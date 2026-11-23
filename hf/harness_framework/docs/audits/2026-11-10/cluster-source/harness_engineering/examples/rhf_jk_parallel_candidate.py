"""Benchmark direct J/K serially and with the CPU task backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys

# Keep the benchmark focused on this module's workers instead of nested BLAS
# threads. These variables must be set before importing NumPy/PySCF.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
import pyscf
from pyscf import gto, lib, scf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hf import Hartree_Fock  # noqa: E402


CASES = [
    (
        "h2-sto3g", "sto-3g",
        "H 0 0 0; H 0 0 1", 5,
    ),
    (
        "water-sto3g", "sto-3g",
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", 5,
    ),
    (
        "water-dimer-sto3g", "sto-3g",
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; "
        "O 2.8 0 0; H 2.8 -0.757 0.587; H 2.8 0.757 0.587", 5,
    ),
    (
        "benzene-631g", "6-31g",
        "C 0.000000 1.396792 0.000000; C -1.209657 0.698396 0.000000; "
        "C -1.209657 -0.698396 0.000000; C 0.000000 -1.396792 0.000000; "
        "C 1.209657 -0.698396 0.000000; C 1.209657 0.698396 0.000000; "
        "H 0.000000 2.481000 0.000000; H -2.147000 1.240500 0.000000; "
        "H -2.147000 -1.240500 0.000000; H 0.000000 -2.481000 0.000000; "
        "H 2.147000 -1.240500 0.000000; H 2.147000 1.240500 0.000000", 1,
    ),
]
WORKERS = (1, 2, 4)
WARMUPS = 1
REPEATS = 5
PARALLEL_CHUNK_SIZE = 16
PARALLEL_MIN_SHELLS = 12


def _timed(function, repeats):
    for _ in range(WARMUPS):
        function()
    samples = []
    import time
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
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
    }


def run_case(name, basis, atom, repeats):
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    dm = scf.RHF(mol).get_init_guess()
    ref_j, ref_k = scf.RHF(mol).get_jk(mol, dm)
    results = {}
    serial_j = serial_k = None
    for workers in WORKERS:
        backend = "serial" if workers == 1 else "cpu"
        solver = Hartree_Fock(
            mol,
            parallel_backend=backend,
            parallel_workers=workers,
            parallel_chunk_size=PARALLEL_CHUNK_SIZE,
            parallel_min_shells=PARALLEL_MIN_SHELLS,
        )
        holder = {}

        def build():
            holder["jk"] = solver.build_jk(dm)

        timing = _timed(build, repeats)
        timing["warmups"] = WARMUPS
        timing["repeats"] = repeats
        current_j, current_k = holder["jk"]
        if workers == 1:
            serial_j, serial_k = current_j, current_k
        ref_error = max(
            float(np.max(np.abs(current_j - ref_j))),
            float(np.max(np.abs(current_k - ref_k))),
        )
        serial_error = 0.0 if workers == 1 else max(
            float(np.max(np.abs(current_j - serial_j))),
            float(np.max(np.abs(current_k - serial_k))),
        )
        results[str(workers)] = {
            "backend": backend,
            "workers": workers,
            "parallel_active": solver.parallel_active,
            "parallel_chunk_size": PARALLEL_CHUNK_SIZE,
            "parallel_min_shells": PARALLEL_MIN_SHELLS,
            "nao": int(mol.nao_nr()),
            "jk_max_abs_vs_pyscf": ref_error,
            "jk_max_abs_vs_serial": serial_error,
            "all_finite": bool(np.isfinite(current_j).all() and np.isfinite(current_k).all()),
            "j_symmetric": bool(np.allclose(current_j, current_j.T, atol=1e-12, rtol=0.0)),
            "k_symmetric": bool(np.allclose(current_k, current_k.T, atol=1e-12, rtol=0.0)),
            "timing": timing,
        }
    return results


def main():
    lib.num_threads(1)
    details = {}
    all_ref_errors = []
    all_parallel_errors = []
    all_finite = True
    serial_medians = []
    worker4_medians = []
    for name, basis, atom, repeats in CASES:
        result = run_case(name, basis, atom, repeats)
        details[name] = result
        for entry in result.values():
            all_ref_errors.append(entry["jk_max_abs_vs_pyscf"])
            all_parallel_errors.append(entry["jk_max_abs_vs_serial"])
            all_finite = all_finite and entry["all_finite"] and entry["j_symmetric"] and entry["k_symmetric"]
        serial_medians.append(result["1"]["timing"]["median_s"])
        worker4_medians.append(result["4"]["timing"]["median_s"])
    serial_median = statistics.median(serial_medians)
    worker4_median = statistics.median(worker4_medians)
    serial_total = sum(serial_medians)
    worker4_total = sum(worker4_medians)
    report = {
        "metrics": {
            "jk_max_abs": max(all_ref_errors),
            "parallel_max_abs": max(all_parallel_errors),
            "all_finite": all_finite,
            "speedup": serial_total / worker4_total,
            "median_seconds": worker4_median,
            "serial_median_seconds": serial_median,
            "worker4_median_seconds": worker4_median,
            "serial_total_median_seconds": serial_total,
            "worker4_total_median_seconds": worker4_total,
            "case_median_speedup": serial_median / worker4_median,
            "case_count": len(CASES),
            "worker_count": len(WORKERS),
        },
        "details": {
            "cases": details,
            "workers": list(WORKERS),
            "warmups": WARMUPS,
            "default_repeats": REPEATS,
            "case_repeats": {name: repeats for name, _, _, repeats in CASES},
            "threads": 1,
            "parallel_chunk_size": PARALLEL_CHUNK_SIZE,
            "parallel_min_shells": PARALLEL_MIN_SHELLS,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyscf": pyscf.__version__,
        },
    }
    report_path = os.environ.get("HARNESS_REPORT_PATH")
    payload = json.dumps(report, indent=2, sort_keys=True)
    if report_path:
        Path(report_path).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
