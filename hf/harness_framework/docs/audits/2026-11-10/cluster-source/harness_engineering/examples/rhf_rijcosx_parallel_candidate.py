"""Benchmark RI-J, COSX, and RIJCOSX-SCF on CPU backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pyscf
import scipy
from pyscf import df, gto, lib, scf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hf import Hartree_Fock  # noqa: E402


CASES = [
    (
        "water-sto3g", "sto-3g",
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", 5, True,
    ),
    (
        "water-dimer-sto3g", "sto-3g",
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; "
        "O 2.8 0 0; H 2.8 -0.757 0.587; H 2.8 0.757 0.587", 5, True,
    ),
    (
        "benzene-631g", "6-31g",
        "C 0.000000 1.396792 0.000000; C -1.209657 0.698396 0.000000; "
        "C -1.209657 -0.698396 0.000000; C 0.000000 -1.396792 0.000000; "
        "C 1.209657 -0.698396 0.000000; C 1.209657 0.698396 0.000000; "
        "H 0.000000 2.481000 0.000000; H -2.147000 1.240500 0.000000; "
        "H -2.147000 -1.240500 0.000000; H 0.000000 -2.481000 0.000000; "
        "H 2.147000 -1.240500 0.000000; H 2.147000 1.240500 0.000000", 1, True,
    ),
]
WORKERS = (1, 2, 4)
WARMUPS = 1
AUXBASIS = "weigend"
DF_TOL = 1e-9
GRID_RAD = 40
GRID_ANG = 146
FINAL_GRID_RAD = 12
FINAL_GRID_ANG = 74
BATCH_SIZE = 256
COSX_TOL = 1e-10
AO_TOL = 1e-10
CHUNK_SIZE = 16
MIN_SHELLS = 12


def timed(function, repeats):
    for _ in range(WARMUPS):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    index = 0.9 * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    p90 = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "minimum_s": min(samples),
        "p90_s": p90,
        "warmups": WARMUPS,
        "repeats": repeats,
    }


def backend_solver(mol, workers):
    return Hartree_Fock(
        mol,
        parallel_backend="serial" if workers == 1 else "cpu",
        parallel_workers=workers,
        parallel_chunk_size=CHUNK_SIZE,
        parallel_min_shells=MIN_SHELLS,
    )


def run_case(name, basis, atom, repeats, run_scf):
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    dm = scf.RHF(mol).get_init_guess()
    ref_df_j, _ = df.DF(mol, auxbasis=AUXBASIS).get_jk(
        dm, with_j=True, with_k=False)
    _, ref_analytic_k = scf.RHF(mol).get_jk(
        mol, dm, with_j=False, with_k=True)
    ref_scf_energy = None
    if run_scf:
        ref_mf = scf.RHF(mol)
        ref_mf.conv_tol = 1e-12
        ref_mf.kernel()
        ref_scf_energy = float(ref_mf.e_tot)
    results = {"ri_j": {}, "cosx_k": {}, "rijcosx_scf": {}}
    serial_j = serial_k = None
    serial_scf_energy = None
    for workers in WORKERS:
        solver = backend_solver(mol, workers)
        holder = {}

        def build_ri_j():
            holder["ri_j"] = solver.build_ri_j(dm, AUXBASIS, df_tol=DF_TOL)

        ri_timing = timed(build_ri_j, repeats)
        ri_j = holder["ri_j"]
        if workers == 1:
            serial_j = ri_j
        results["ri_j"][str(workers)] = {
            "workers": workers,
            "backend": solver.parallel_backend,
            "parallel_active": solver.parallel_active,
            "nao": int(mol.nao_nr()),
            "matrix_error_vs_pyscf_df": float(np.max(np.abs(ri_j - ref_df_j))),
            "matrix_error_vs_serial": 0.0 if workers == 1 else float(np.max(np.abs(ri_j - serial_j))),
            "all_finite": bool(np.isfinite(ri_j).all()),
            "symmetric": bool(np.allclose(ri_j, ri_j.T, atol=1e-12, rtol=0.0)),
            "timing": ri_timing,
        }

        def build_cosx_k():
            holder["cosx_k"] = solver.build_cosx_k(
                dm, grid_rad=GRID_RAD, grid_ang=GRID_ANG,
                batch_size=BATCH_SIZE, cosx_tol=COSX_TOL,
                ao_tol=AO_TOL)

        cosx_timing = timed(build_cosx_k, repeats)
        cosx_k = holder["cosx_k"]
        if workers == 1:
            serial_k = cosx_k
        results["cosx_k"][str(workers)] = {
            "workers": workers,
            "backend": solver.parallel_backend,
            "parallel_active": solver.parallel_active,
            "nao": int(mol.nao_nr()),
            "matrix_error_vs_pyscf_analytic": float(np.max(np.abs(cosx_k - ref_analytic_k))),
            "matrix_error_vs_serial": 0.0 if workers == 1 else float(np.max(np.abs(cosx_k - serial_k))),
            "all_finite": bool(np.isfinite(cosx_k).all()),
            "symmetric": bool(np.allclose(cosx_k, cosx_k.T, atol=1e-10, rtol=0.0)),
            "timing": cosx_timing,
        }

        if run_scf:
            hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            overlap = mol.intor("int1e_ovlp")
            scf_holder = {}

            def solve():
                scf_holder["result"] = solver.solve_rhf(
                    mol.nelec, hcore, overlap, mol, mol.energy_nuc(),
                    tol=1e-8, jk_mode="rijcosx", auxbasis=AUXBASIS,
                    df_tol=DF_TOL, cosx_tol=COSX_TOL, ao_tol=AO_TOL,
                    grid_rad=GRID_RAD, grid_ang=GRID_ANG,
                    final_grid_rad=FINAL_GRID_RAD, final_grid_ang=FINAL_GRID_ANG,
                    batch_size=BATCH_SIZE)

            scf_timing = timed(solve, 1)
            energy, density = scf_holder["result"]
            if workers == 1:
                serial_scf_energy = energy
            results["rijcosx_scf"][str(workers)] = {
                "workers": workers,
                "backend": solver.parallel_backend,
                "parallel_active": solver.parallel_active,
                "energy_hartree": float(energy),
                "energy_error_vs_pyscf": float(abs(energy - ref_scf_energy)),
                "energy_error_vs_serial": 0.0 if workers == 1 else float(abs(energy - serial_scf_energy)),
                "scf_converged": True,
                "density_finite": bool(np.isfinite(density).all()),
                "density_shape": list(density.shape),
                "timing": scf_timing,
            }
    return {"basis": basis, "nao": int(mol.nao_nr()), "run_scf": run_scf, "modules": results}


def summarize(details):
    ri_ref_errors = []
    ri_parallel_errors = []
    cosx_ref_errors = []
    cosx_parallel_errors = []
    all_finite = True
    ri_serial = []
    ri_worker4 = []
    cosx_serial = []
    cosx_worker4 = []
    scf_serial = []
    scf_worker4 = []
    scf_parallel_errors = []
    scf_reference_errors = []
    scf_converged = True
    scf_finite = True
    for case in details.values():
        ri = case["modules"]["ri_j"]
        cosx = case["modules"]["cosx_k"]
        ri_ref_errors.extend(x["matrix_error_vs_pyscf_df"] for x in ri.values())
        ri_parallel_errors.extend(x["matrix_error_vs_serial"] for x in ri.values())
        cosx_ref_errors.extend(x["matrix_error_vs_pyscf_analytic"] for x in cosx.values())
        cosx_parallel_errors.extend(x["matrix_error_vs_serial"] for x in cosx.values())
        all_finite = all_finite and all(x["all_finite"] and x["symmetric"] for x in ri.values())
        all_finite = all_finite and all(x["all_finite"] and x["symmetric"] for x in cosx.values())
        ri_serial.append(ri["1"]["timing"]["median_s"])
        ri_worker4.append(ri["4"]["timing"]["median_s"])
        cosx_serial.append(cosx["1"]["timing"]["median_s"])
        cosx_worker4.append(cosx["4"]["timing"]["median_s"])
        if case["run_scf"]:
            scf = case["modules"]["rijcosx_scf"]
            scf_serial.append(scf["1"]["timing"]["median_s"])
            scf_worker4.append(scf["4"]["timing"]["median_s"])
            scf_parallel_errors.extend(x["energy_error_vs_serial"] for x in scf.values())
            scf_reference_errors.extend(x["energy_error_vs_pyscf"] for x in scf.values())
            scf_converged = scf_converged and all(x["scf_converged"] for x in scf.values())
            scf_finite = scf_finite and all(x["density_finite"] for x in scf.values())
    ri_speedup = sum(ri_serial) / sum(ri_worker4)
    cosx_speedup = sum(cosx_serial) / sum(cosx_worker4)
    scf_speedup = (sum(scf_serial) / sum(scf_worker4)) if scf_serial else 1.0
    return {
        "ri_j_max_abs": max(ri_ref_errors),
        "ri_j_parallel_max_abs": max(ri_parallel_errors),
        "cosx_k_max_abs": max(cosx_ref_errors),
        "cosx_k_parallel_max_abs": max(cosx_parallel_errors),
        "scf_parallel_energy_max_abs": max(scf_parallel_errors) if scf_parallel_errors else 0.0,
        "scf_pyscf_energy_max_abs": max(scf_reference_errors) if scf_reference_errors else 0.0,
        "all_finite": all_finite and scf_finite,
        "scf_converged": scf_converged,
        "ri_j_speedup": ri_speedup,
        "cosx_k_speedup": cosx_speedup,
        "rijcosx_scf_speedup": scf_speedup,
        "speedup": min(ri_speedup, cosx_speedup, scf_speedup),
        "ri_j_serial_total_median_seconds": sum(ri_serial),
        "ri_j_worker4_total_median_seconds": sum(ri_worker4),
        "cosx_k_serial_total_median_seconds": sum(cosx_serial),
        "cosx_k_worker4_total_median_seconds": sum(cosx_worker4),
        "rijcosx_scf_serial_total_median_seconds": sum(scf_serial),
        "rijcosx_scf_worker4_total_median_seconds": sum(scf_worker4),
    }


def main():
    lib.num_threads(1)
    details = {}
    for name, basis, atom, repeats, run_scf in CASES:
        details[name] = run_case(name, basis, atom, repeats, run_scf)
    report = {
        "metrics": summarize(details),
        "details": {
            "cases": details,
            "workers": list(WORKERS),
            "auxbasis": AUXBASIS,
            "df_tol": DF_TOL,
            "grid_rad": GRID_RAD,
            "grid_ang": GRID_ANG,
            "final_grid_rad": FINAL_GRID_RAD,
            "final_grid_ang": FINAL_GRID_ANG,
            "batch_size": BATCH_SIZE,
            "parallel_chunk_size": CHUNK_SIZE,
            "parallel_min_shells": MIN_SHELLS,
            "threads": 1,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyscf": pyscf.__version__,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    report_path = os.environ.get("HARNESS_REPORT_PATH")
    if report_path:
        Path(report_path).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
