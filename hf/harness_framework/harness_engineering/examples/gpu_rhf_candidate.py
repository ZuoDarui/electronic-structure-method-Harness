"""Independent numerical checks for the GPU friendly, self-written RHF solver.

The task file supplies cases and thresholds. PySCF's SCF algorithm is used only
as an external reference; it is never called by the candidate solver. CPU and
GPU verification have separate task files, and a GPU request cannot fall back.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from pyscf import gto, lib, scf

ROOT = Path(__file__).resolve().parents[2]
# A direct script must select this framework before importing its package.
for _module_name, _filename in (("harness_engineering", "__init__.py"),
                                ("harness_engineering.paths", "paths.py")):
    _cached = sys.modules.get(_module_name)
    if _cached is not None:
        _origin = getattr(_cached, "__file__", None)
        if _origin is None or Path(_origin).resolve() != ROOT / "harness_engineering" / _filename:
            raise ImportError(f"{_module_name} is already loaded outside this harness framework")
sys.path[:] = [str(ROOT), *(value for value in sys.path if value != str(ROOT))]
from harness_engineering.paths import HF_ROOT, configure_import_paths  # noqa: E402

configure_import_paths()

from gpu_hf import GPUFriendlyRHF  # noqa: E402


def _max_abs(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _versions() -> dict:
    result = {"python": sys.version, "platform": platform.platform(),
              "machine": platform.machine()}
    for package in ("numpy", "scipy", "pyscf", "cupy", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _reference(mol, mode: str, auxbasis: str):
    reference = scf.RHF(mol)
    if mode == "df":
        reference = reference.density_fit(auxbasis=auxbasis)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 150
    return reference


def _case(case: dict, mode: str, backend: str, auxbasis: str,
          matrix_dir: Path | None, seed: int) -> dict:
    name = f"{case['name']}-{mode}-{backend}"
    print(f"START {name}", flush=True)
    mol = gto.M(atom=case["atom"], basis=case["basis"],
                unit=case.get("unit", "Angstrom"), charge=case.get("charge", 0),
                spin=0, verbose=0)

    def progress(row):
        print(json.dumps({"case": name, "iteration": row}, sort_keys=True), flush=True)

    started = time.perf_counter()
    solver = GPUFriendlyRHF(mol, backend=backend, integral_mode=mode,
                           auxbasis=auxbasis, conv_tol=1e-11,
                           residual_tol=1e-9, max_cycle=150, callback=progress)
    nao = mol.nao_nr()
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(2, nao, nao)) * 0.2
    trial = (trial + trial.transpose(0, 2, 1)) / 2
    reference = _reference(mol, mode, auxbasis)
    j_backend, k_backend = solver.build_jk(trial[0])
    gpu_array = type(j_backend).__module__.split(".")[0] == "cupy"
    solver.synchronize()
    j, k = solver.to_numpy(j_backend), solver.to_numpy(k_backend)
    ref_j, ref_k = reference.get_jk(mol, trial[0])
    j2, k2 = map(solver.to_numpy, solver.build_jk(trial[1]))
    jsum, ksum = map(solver.to_numpy, solver.build_jk(trial[0] + trial[1]))
    result = solver.kernel()
    solver.synchronize()
    candidate_seconds = time.perf_counter() - started

    reference_started = time.perf_counter()
    reference.kernel()
    reference_seconds = time.perf_counter() - reference_started
    dm = np.asarray(result.density)
    coeff = np.asarray(result.mo_coeff)
    eps = np.asarray(result.mo_energy)
    occupations = np.asarray(result.orbital_occupations)
    overlap = mol.intor("int1e_ovlp")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    # Reconstruct final Fock independently of the solver's recorded residual.
    final_j, final_k = reference.get_jk(mol, dm)
    fock = hcore + final_j - 0.5 * final_k
    reference_dm = reference.make_rdm1()
    recomputed_energy = float(0.5 * np.einsum("ij,ji", dm, hcore + fock) + mol.energy_nuc())
    reconstructed_dm = (coeff * occupations) @ coeff.T
    errors = {
        "energy_max_abs": abs(float(result.energy) - float(reference.e_tot)),
        "jk_max_abs": max(_max_abs(j - ref_j), _max_abs(k - ref_k)),
        "jk_symmetry_max_abs": max(_max_abs(j - j.T), _max_abs(k - k.T)),
        "jk_linearity_max_abs": max(_max_abs(jsum - j - j2), _max_abs(ksum - k - k2)),
        "density_max_abs": _max_abs(dm - reference_dm),
        "electron_count_abs": abs(float(np.trace(dm @ overlap)) - mol.nelectron),
        "idempotency_max_abs": _max_abs(dm @ overlap @ dm - 2 * dm),
        "residual_max_abs": _max_abs(fock @ dm @ overlap - overlap @ dm @ fock),
        "orthogonality_max_abs": _max_abs(coeff.T @ overlap @ coeff - np.eye(coeff.shape[1])),
        "orbital_equation_max_abs": _max_abs(fock @ coeff - (overlap @ coeff) * eps),
        "density_reconstruction_max_abs": _max_abs(dm - reconstructed_dm),
        "energy_reconstruction_abs": abs(float(result.energy) - recomputed_energy),
        "anchor_energy_abs": abs(float(result.energy) - float(case["anchor_energy"]))
                             if "anchor_energy" in case and mode == "exact" else 0.0,
    }
    arrays = (dm, coeff, eps, occupations, j, k, fock, overlap,
              reference_dm, ref_j, ref_k)
    all_finite = (all(bool(np.isfinite(a).all()) for a in arrays)
                  and all(bool(np.isfinite(v)) for v in errors.values())
                  and bool(np.isfinite(result.energy))
                  and bool(np.isfinite(reference.e_tot)))
    history = result.history
    all_finite = all_finite and all(
        np.isfinite(value) for row in history for key, value in row.items()
        if isinstance(value, (int, float))
    )
    detail = {
        "name": name, "input": case, "integral_mode": mode, "backend": result.backend,
        "auxbasis": auxbasis if mode == "df" else None, "nao": nao,
        "candidate_energy": float(result.energy), "reference_energy": float(reference.e_tot),
        "candidate_converged": bool(result.converged), "reference_converged": bool(reference.converged),
        "all_finite": bool(all_finite), "gpu_array_observed": gpu_array,
        "candidate_seconds": candidate_seconds, "reference_seconds": reference_seconds,
        "iterations": result.iterations, "history": history, "errors": errors,
        "solver_metadata": result.metadata,
    }
    if matrix_dir is not None:
        path = matrix_dir / f"{name}.npz"
        np.savez_compressed(path, density=dm, reference_density=reference_dm,
                            mo_coeff=coeff, mo_energy=eps, occupations=occupations,
                            overlap=overlap, hcore=hcore, fock=fock, trial_density=trial[0],
                            j=j, k=k, reference_j=ref_j, reference_k=ref_k)
        detail["matrix_artifact"] = str(path)
        detail["matrix_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"DONE {name}: energy={result.energy:.12f} "
          f"|dE|={errors['energy_max_abs']:.3e} "
          f"|dJK|={errors['jk_max_abs']:.3e} "
          f"converged={result.converged}/{reference.converged}", flush=True)
    return detail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args(argv)
    task_path = args.task.resolve()
    spec = json.loads(task_path.read_text())
    backend = spec.get("env", {}).get("GPU_HF_BACKEND", "numpy")
    modes = spec.get("validation", {}).get("integral_modes", ["exact", "df"])
    auxbasis = spec.get("validation", {}).get("auxbasis", "weigend")
    random_seed = int(spec.get("validation", {}).get("random_seed", 20261110))
    report_path = Path(os.environ.get("HARNESS_REPORT_PATH", f"{spec['name']}.report.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Derive evidence paths from the harness's unique report name. A mutation
    # experiment may reuse the input task while needing its own matrix archive.
    matrix_dir = report_path.parent / f"{report_path.name.removesuffix('.report.json')}.matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(1)
    details = {"environment": _versions(), "requested_backend": backend,
               "random_seed": random_seed,
               "integral_modes": modes, "cases": [], "sources_sha256": {},
               "timing_scope": "Candidate time includes preparation, J/K property checks and SCF; reference time is SCF only. These timings are diagnostics and are not a speedup benchmark.",
               "reference_scope": "SCF, DIIS and matrix contractions are checked against PySCF SCF. Both paths share PySCF's integral engine and basis data, so this does not independently certify the integral engine."}
    for path in (HF_ROOT / "gpu_hf.py", Path(__file__).resolve(), task_path):
        details["sources_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    reasons = []
    try:
        for index, case in enumerate(spec["cases"]):
            for mode in modes:
                details["cases"].append(_case(case, mode, backend, auxbasis, matrix_dir, random_seed + index))
        details["df_approximation_diagnostics"] = []
        for case in spec["cases"]:
            records = {record["integral_mode"]: record for record in details["cases"]
                       if record["input"]["name"] == case["name"]}
            if "exact" in records and "df" in records:
                details["df_approximation_diagnostics"].append({
                    "case": case["name"],
                    "df_vs_exact_candidate_energy_abs": abs(records["df"]["candidate_energy"] - records["exact"]["candidate_energy"]),
                    "df_vs_exact_reference_energy_abs": abs(records["df"]["reference_energy"] - records["exact"]["reference_energy"]),
                    "note": "Density fitting changes the numerical approximation; this diagnostic is separate from the error relative to the matching method reference.",
                })
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        details["exception_traceback"] = traceback.format_exc()
        print(details["exception_traceback"], file=sys.stderr, flush=True)
    completed = details["cases"]
    expected_count = len(spec["cases"]) * len(modes)
    if len(completed) != expected_count:
        reasons.append(f"completed {len(completed)} of {expected_count} required case/mode pairs")
    metrics = {
        key: max((case["errors"][key] for case in completed), default=float("1e300"))
        for key in spec["correctness_tolerances"]
    }
    metrics.update({
        "case_count": len(completed), "all_finite": bool(completed) and all(c["all_finite"] for c in completed),
        "scf_converged": bool(completed) and all(c["candidate_converged"] and c["reference_converged"] for c in completed),
        "gpu_executed": bool(completed) and all(c["gpu_array_observed"] and c["backend"] == "cupy" for c in completed),
        "candidate_total_seconds": sum(c["candidate_seconds"] for c in completed),
        "reference_total_seconds": sum(c["reference_seconds"] for c in completed),
    })
    if backend == "cupy" and not metrics["gpu_executed"]:
        reasons.append("GPU execution required but CuPy device arrays were not observed for every case")
    for key, tolerance in spec["correctness_tolerances"].items():
        value = metrics[key]
        if not np.isfinite(value) or abs(value) > tolerance:
            reasons.append(f"{key}={value} exceeds tolerance {tolerance}")
    if not metrics["all_finite"]:
        reasons.append("non-finite output or no completed case")
    if not metrics["scf_converged"]:
        reasons.append("candidate or reference did not converge, or no completed case")
    report = {"metrics": metrics, "details": details, "passed": not reasons, "reasons": reasons}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metrics": metrics, "passed": not reasons, "reasons": reasons}, indent=2), flush=True)
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
