"""Child process that times ``solve(mol)`` for one submitted job.

The candidate's working directory contains only its source and the job file.
Reference energies stay in the parent process.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def _block_drivers(prefixes: list[str]) -> None:
    """PySCF imports ``scf`` during package init, so drivers are replaced in place.

    ``pyscf.gto`` integrals stay available. A generated program that still
    reaches ``RHF`` or ``DF`` fails at the call.
    """
    if not prefixes:
        return

    def blocked(*_args, **_kwargs):
        raise RuntimeError("pyscf SCF and density-fitting drivers are blocked")

    if "pyscf.scf" in prefixes:
        for mod_name in ("pyscf.scf", "pyscf.scf.hf", "pyscf.scf.uhf", "pyscf.scf.rohf"):
            module = importlib.import_module(mod_name)
            for name in ("RHF", "UHF", "ROHF", "GHF", "RKS", "UKS"):
                if hasattr(module, name):
                    setattr(module, name, blocked)
    if "pyscf.df" in prefixes:
        module = importlib.import_module("pyscf.df")
        if hasattr(module, "DF"):
            setattr(module, "DF", blocked)


def _limit_memory(memory_mb: int | None) -> None:
    if not memory_mb:
        return
    try:
        import resource

        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        return


def _load_solve(program: Path):
    spec = importlib.util.spec_from_file_location("esm_candidate", program)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise AttributeError("candidate has no callable solve(mol)")
    return solve


def _build_mol(case: dict):
    from pyscf import gto

    return gto.M(
        atom=case["atom"],
        basis=case["basis"],
        unit=case.get("unit", "Angstrom"),
        charge=int(case.get("charge", 0)),
        spin=int(case.get("spin", 0)),
        verbose=0,
    )


def _time_call(program: Path, case: dict, repeats: int):
    """Each call gets a freshly loaded module and a fresh Mole, so state kept in
    module globals or on the molecule cannot carry over between timed calls."""
    _load_solve(program)(_build_mol(case))
    samples = []
    result = None
    for _ in range(repeats):
        solve = _load_solve(program)
        mol = _build_mol(case)
        started = time.perf_counter()
        result = solve(mol)
        samples.append(time.perf_counter() - started)
    samples.sort()
    return result, samples[len(samples) // 2]


def run_job(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = Path(job["output"])
    if not output.is_absolute():
        output = job_path.parent / output
    payload: dict = {"status": "error", "stderr": "", "cases": []}
    try:
        _block_drivers(list(job.get("block_imports") or []))
        _limit_memory(job.get("memory_mb"))
        program = Path(job["program"])
        _load_solve(program)
        repeats = int(job.get("repeats", 3))
        cases = []
        for case in job["cases"]:
            captured = []
            stdout = sys.stdout
            sys.stdout = _Collector(captured)
            try:
                result, median = _time_call(program, case, repeats)
            finally:
                sys.stdout = stdout
            if not isinstance(result, dict) or "e_tot" not in result:
                raise TypeError(f"{case['id']} solve() did not return e_tot")
            energy = float(result["e_tot"])
            if energy != energy:
                raise ValueError(f"{case['id']} energy is not finite")
            cases.append(
                {
                    "id": case["id"],
                    "energy": energy,
                    "converged": bool(result.get("converged", False)),
                    "time_s": float(median),
                    "stdout": "".join(captured)[-4000:],
                }
            )
        payload = {"status": "ok", "stderr": "", "cases": cases}
    except Exception:
        payload["stderr"] = traceback.format_exc()[-16000:]
        payload["status"] = "error"
    output.write_text(json.dumps(payload), encoding="utf-8")


class _Collector:
    def __init__(self, sink: list[str]):
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    run_job(Path(args.job))


if __name__ == "__main__":
    main()
