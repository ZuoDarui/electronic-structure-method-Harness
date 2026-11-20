"""Child process that times ``solve(mol)`` for one submitted job.

The candidate's working directory contains only its source and the job file.
Reference energies stay in the parent process.

Everything the job does goes to standard output, which the cluster script
sends to ``worker.log``: each case's start and end, a heartbeat every
``heartbeat_s`` seconds, whatever the program prints, and one line per SCF
cycle that the program reports through the ``log`` callback it is given as
``solve(mol, method, log=log)``. The same cycles are returned in result.json.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path

MAX_ITERATIONS_KEPT = 2000

_OUT = sys.stdout
_OUT_LOCK = threading.Lock()


def _emit(text: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _OUT_LOCK:
        for line in text.splitlines() or [""]:
            _OUT.write(f"[{stamp}] {line}\n")
        _OUT.flush()


def _max_rss_gb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except (ImportError, OSError):
        return None


def _number(value, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"log(): {name} is not finite ({value!r})")
    return number


class IterationLog:
    """The ``log`` callback handed to ``solve``: one call per SCF cycle.

    The wall time of each cycle is measured here, between successive calls,
    so it does not depend on the program's own timing.
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.records: list[dict] = []
        self.count = 0
        self.restart()

    def restart(self) -> float:
        self.started = self._last = time.perf_counter()
        return self.started

    def __call__(self, cycle, energy, delta_e=None, grad_norm=None, stage: str = "scf", **extra) -> None:
        now = time.perf_counter()
        record = {
            "stage": str(stage),
            "cycle": int(cycle),
            "energy": _number(energy, "energy"),
            "delta_e": _number(delta_e, "delta_e"),
            "grad_norm": _number(grad_norm, "grad_norm"),
            "cycle_time_s": now - self._last,
            "elapsed_s": now - self.started,
        }
        if record["energy"] is None:
            raise ValueError("log(): energy is required")
        self._last = now
        self.count += 1
        if len(self.records) < MAX_ITERATIONS_KEPT:
            self.records.append(record)
        else:
            self.records[-1] = record
        notes = " ".join(f"{key}={value}" for key, value in extra.items())
        _emit(
            f"{self.case_id} {record['stage']} cycle {record['cycle']}: E={record['energy']:.12f}"
            f" dE={_fmt(record['delta_e'])} |g|={_fmt(record['grad_norm'])}"
            f" cycle_time={record['cycle_time_s']:.2f}s elapsed={record['elapsed_s']:.2f}s"
            + (f" {notes}" if notes else "")
        )

    @property
    def last(self) -> dict | None:
        return self.records[-1] if self.records else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3e}"


class _Heartbeat:
    """Writes a line every ``interval`` seconds while a case runs."""

    def __init__(self, case_id: str, interval: float, iterations: IterationLog):
        self.case_id, self.interval, self.iterations = case_id, interval, iterations
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if self.interval > 0:
            self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            last = self.iterations.last
            rss = _max_rss_gb()
            _emit(
                f"{self.case_id} running: elapsed={time.perf_counter() - self.iterations.started:.0f}s"
                f" cycles_logged={self.iterations.count}"
                + (f" last={last['stage']} cycle {last['cycle']} E={last['energy']:.12f}" if last else "")
                + (f" max_rss={rss:.1f}GB" if rss is not None else "")
            )


def _block_drivers(prefixes: list[str]) -> None:
    """PySCF imports ``scf`` during package init, so drivers are replaced in place.

    ``pyscf.gto`` integrals stay available. A generated program that still
    reaches ``RHF`` or ``DF`` fails at the call.
    """
    if not prefixes:
        return

    def blocked(*_args, **_kwargs):
        raise RuntimeError("pyscf SCF and density-fitting drivers are blocked")

    # These define classes that subclass scf.hf.RHF, so they must be imported
    # before RHF is replaced.
    for mod_name in ("pyscf.dft", "pyscf.df.df_jk", "pyscf.sgx"):
        importlib.import_module(mod_name)

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
        df_jk = importlib.import_module("pyscf.df.df_jk")
        for name in ("get_j", "get_jk", "density_fit"):
            if hasattr(df_jk, name):
                setattr(df_jk, name, blocked)
    if "pyscf.sgx" in prefixes:
        for mod_name, names in (
            ("pyscf.sgx", ("SGX", "sgx_fit")),
            ("pyscf.sgx.sgx", ("SGX", "sgx_fit")),
            ("pyscf.sgx.sgx_jk", ("get_jk", "get_k_only", "get_jk_favork", "get_jk_favorj")),
        ):
            module = importlib.import_module(mod_name)
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, blocked)


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

    mol = gto.M(
        atom=case["atom"],
        basis=case["basis"],
        unit=case.get("unit", "Angstrom"),
        charge=int(case.get("charge", 0)),
        spin=int(case.get("spin", 0)),
        verbose=0,
    )
    if case.get("nao") is not None and mol.nao_nr() != int(case["nao"]):
        raise ValueError(f"{case['id']}: built {mol.nao_nr()} basis functions, suite says {case['nao']}")
    return mol


def accepts_log(solve) -> bool:
    try:
        parameters = inspect.signature(solve).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "log" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _call_solve(solve, case: dict, mol=None, log: IterationLog | None = None):
    """``solve(mol)``, or ``solve(mol, method)`` for suites that fix the method;
    ``log`` is passed as a keyword to programs that take it."""
    mol = _build_mol(case) if mol is None else mol
    kwargs = {"log": log} if log is not None and accepts_log(solve) else {}
    if case.get("method") is not None:
        return solve(mol, dict(case["method"]), **kwargs)
    return solve(mol, **kwargs)


def _time_call(program: Path, case: dict, repeats: int, warmup: bool = True, heartbeat_s: float = 0):
    """Each call gets a freshly loaded module and a fresh Mole, so state kept in
    module globals or on the molecule cannot carry over between timed calls.
    Returns the last call's result and cycle log, and the median time."""
    if warmup:
        _call_solve(_load_solve(program), case, log=IterationLog(f"{case['id']} (warmup)"))
    samples = []
    result = None
    iterations = None
    for _ in range(repeats):
        solve = _load_solve(program)
        mol = _build_mol(case)
        iterations = IterationLog(case["id"])
        with _Heartbeat(case["id"], heartbeat_s, iterations):
            started = iterations.restart()
            result = _call_solve(solve, case, mol, iterations)
            samples.append(time.perf_counter() - started)
    samples.sort()
    return result, samples[len(samples) // 2], iterations


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
        solve = _load_solve(program)
        require_log = bool(job.get("require_iteration_log"))
        if require_log and not accepts_log(solve):
            raise TypeError("solve() must take a log argument: solve(mol, method, log)")
        repeats = int(job.get("repeats", 3))
        warmup = bool(job.get("warmup", True))
        heartbeat_s = float(job.get("heartbeat_s") or 0)
        cases = []
        for case in job["cases"]:
            _emit(f"{case['id']} start: {case.get('nao', '?')} basis functions, method {case.get('method')}")
            captured = []
            stdout = sys.stdout
            sys.stdout = _Collector(captured)
            try:
                result, median, iterations = _time_call(program, case, repeats, warmup, heartbeat_s)
            finally:
                sys.stdout = stdout
            if not isinstance(result, dict) or "e_tot" not in result:
                raise TypeError(f"{case['id']} solve() did not return e_tot")
            energy = float(result["e_tot"])
            if energy != energy:
                raise ValueError(f"{case['id']} energy is not finite")
            if require_log and not iterations.count:
                raise RuntimeError(f"{case['id']}: solve() returned without calling log() for any SCF cycle")
            converged = bool(result.get("converged", False))
            _emit(
                f"{case['id']} done: E={energy:.12f} converged={converged} time={median:.2f}s"
                f" cycles_logged={iterations.count}"
            )
            cases.append(
                {
                    "id": case["id"],
                    "energy": energy,
                    "converged": converged,
                    "time_s": float(median),
                    "scf_cycles": iterations.count,
                    "iterations": iterations.records,
                    "stdout": "".join(captured)[-4000:],
                }
            )
        payload = {"status": "ok", "stderr": "", "cases": cases}
    except Exception:
        payload["stderr"] = traceback.format_exc()[-16000:]
        payload["status"] = "error"
        _emit("job failed:\n" + payload["stderr"])
    output.write_text(json.dumps(payload), encoding="utf-8")


class _Collector:
    """Keeps what the program prints for result.json and passes it on to the log."""

    def __init__(self, sink: list[str]):
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.append(text)
        with _OUT_LOCK:
            _OUT.write(text)
        return len(text)

    def flush(self) -> None:
        with _OUT_LOCK:
            _OUT.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    run_job(Path(args.job))


if __name__ == "__main__":
    main()
