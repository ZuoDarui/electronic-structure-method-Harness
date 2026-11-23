"""Time one RIJCOSX get_jk on the water-cluster scale ladder.

16 waters always runs, serially and with four CPU workers. Larger clusters
use the same path, but are skipped when their estimated wall time exceeds
the time still left in the task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from pyscf import df, gto, lib, scf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hf import Hartree_Fock, estimate_dense_solver_memory_gib  # noqa: E402
from harness_engineering.water_clusters import (  # noqa: E402
    CLUSTER_COUNTS,
    cluster_atom_string,
    nao_mismatch_max,
)


BASIS = "cc-pvtz"
AUXBASIS = "weigend"
DF_TOL = 1e-9
GRID_RAD = 30
GRID_ANG = 110
MONOMER_GRID_RAD = 40
MONOMER_GRID_ANG = 146
BATCH_SIZE = 256
COSX_TOL = 1e-10
AO_TOL = 1e-10
WORKERS = 4
CHUNK_SIZE = 16
MIN_SHELLS = 12
TASK_TIMEOUT_S = 3600
MONOMER_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"


def estimate_jk_seconds(previous_seconds, previous_cost, next_cost):
    """Scale a finished get_jk time by significant pairs times atom count."""

    if previous_seconds is None or previous_seconds <= 0 or previous_cost <= 0:
        return None
    return float(previous_seconds) * (float(next_cost) / float(previous_cost))


def should_skip_jk(estimated_seconds, remaining_seconds):
    return estimated_seconds is not None and estimated_seconds > remaining_seconds


def peak_rss_gib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)
    return rss / (1024 ** 2) / 1024


def _asymmetry(matrix):
    return float(np.max(np.abs(matrix - matrix.T)))


def _finite(matrix):
    return bool(np.isfinite(matrix).all())


def make_solver(mol, workers):
    if workers == 1:
        return Hartree_Fock(mol, parallel_backend="serial")
    return Hartree_Fock(
        mol,
        parallel_backend="cpu",
        parallel_workers=workers,
        parallel_chunk_size=CHUNK_SIZE,
        parallel_min_shells=MIN_SHELLS,
    )


def run_get_jk(solver, dm):
    started = time.perf_counter()
    j_matrix, k_matrix = solver.get_jk(
        dm,
        jk_mode="rijcosx",
        auxbasis=AUXBASIS,
        df_tol=DF_TOL,
        grid_rad=GRID_RAD,
        grid_ang=GRID_ANG,
        batch_size=BATCH_SIZE,
        cosx_tol=COSX_TOL,
        ao_tol=AO_TOL,
        overlap_fit=True,
    )
    return j_matrix, k_matrix, time.perf_counter() - started


def cluster_cost(solver, natm):
    return max(int(solver.significant_shell_pairs), 1) * int(natm)


def monomer_cosx():
    mol = gto.M(atom=MONOMER_ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    dm = np.asarray(scf.hf.init_guess_by_minao(mol))
    solver = make_solver(mol, 1)
    k_matrix = solver.build_cosx_k(
        dm,
        grid_rad=MONOMER_GRID_RAD,
        grid_ang=MONOMER_GRID_ANG,
        batch_size=BATCH_SIZE,
        cosx_tol=COSX_TOL,
        ao_tol=AO_TOL,
        overlap_fit=True,
    )
    _, reference = scf.RHF(mol).get_jk(mol, dm, with_j=False, with_k=True)
    return {
        "nao": int(mol.nao_nr()),
        "nshells": int(mol.nbas),
        "grid_rad": MONOMER_GRID_RAD,
        "grid_ang": MONOMER_GRID_ANG,
        "cosx_max_abs": float(np.max(np.abs(k_matrix - reference))),
        "cosx_full_tensor_calls": int(solver.cosx_full_tensor_calls),
        "all_finite": _finite(k_matrix) and _finite(reference),
    }


def prepare_cluster(count, monomer_nao):
    mol = gto.M(
        atom=cluster_atom_string(count),
        basis=BASIS,
        unit="Angstrom",
        verbose=0,
    )
    dm = np.asarray(scf.hf.init_guess_by_minao(mol))
    solver = make_solver(mol, WORKERS)
    memory = estimate_dense_solver_memory_gib(mol.nao_nr())
    return {
        "mol": mol,
        "dm": dm,
        "solver": solver,
        "molecules": count,
        "basis": BASIS,
        "natm": int(mol.natm),
        "nao": int(mol.nao_nr()),
        "expected_nao": int(monomer_nao) * count,
        "nshells": int(solver.nshells),
        "nelectron": int(mol.nelectron),
        "significant_shell_pairs": int(solver.significant_shell_pairs),
        "significant_pair_fraction": float(solver.significant_pair_fraction),
        "estimated_dense_working_set_gib": memory["estimated_dense_working_set_gib"],
        "cost": cluster_cost(solver, mol.natm),
    }


def main():
    lib.num_threads(1)
    started = time.perf_counter()
    monomer = monomer_cosx()
    finite = bool(monomer["all_finite"])
    clusters = {}
    previous = None
    errors = {}
    for count in CLUSTER_COUNTS:
        prepared = prepare_cluster(count, monomer["nao"])
        record = {
            key: value
            for key, value in prepared.items()
            if key not in ("mol", "dm", "solver")
        }
        record["skipped"] = False
        record["parallel_workers"] = WORKERS
        remaining = TASK_TIMEOUT_S - (time.perf_counter() - started)
        estimate = None
        if previous is not None:
            estimate = estimate_jk_seconds(
                previous["seconds"], previous["cost"], prepared["cost"])
        if count != 16 and should_skip_jk(estimate, remaining):
            record["skipped"] = True
            record["estimated_seconds"] = estimate
            record["skip_reason"] = (
                f"estimated get_jk time {estimate:.3f} s exceeds "
                f"the remaining {remaining:.3f} s"
            )
            print(record["skip_reason"], file=sys.stderr, flush=True)
            clusters[str(count)] = record
            continue
        print(
            f"get_jk molecules={count} pairs={prepared['significant_shell_pairs']} "
            f"remaining={remaining:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if count == 16:
            serial = make_solver(prepared["mol"], 1)
            j_serial, k_serial, serial_seconds = run_get_jk(serial, prepared["dm"])
            reference, _ = df.DF(prepared["mol"], auxbasis=AUXBASIS).get_jk(
                prepared["dm"], with_j=True, with_k=False)
            errors["ri_j_16_max_abs"] = float(np.max(np.abs(j_serial - reference)))
            errors["j_asymmetry_16"] = _asymmetry(j_serial)
            errors["k_asymmetry_16"] = _asymmetry(k_serial)
            finite = finite and _finite(j_serial) and _finite(k_serial)
            record["serial_seconds"] = serial_seconds
        j_matrix, k_matrix, elapsed = run_get_jk(prepared["solver"], prepared["dm"])
        finite = finite and _finite(j_matrix) and _finite(k_matrix)
        record["jk_seconds"] = elapsed
        record["rss_gib"] = peak_rss_gib()
        record["ri_screened_pairs"] = int(prepared["solver"].last_ri_screened_pairs)
        record["cosx_full_tensor_calls"] = int(prepared["solver"].cosx_full_tensor_calls)
        if count == 16:
            errors["j_parallel_16_max_abs"] = float(np.max(np.abs(j_matrix - j_serial)))
            errors["k_parallel_16_max_abs"] = float(np.max(np.abs(k_matrix - k_serial)))
        clusters[str(count)] = record
        previous = {"seconds": elapsed, "cost": prepared["cost"]}

    nao_by_count = {count: clusters[str(count)]["nao"] for count in CLUSTER_COUNTS}
    metrics = {
        "monomer_nao": monomer["nao"],
        "nao_mismatch_max": nao_mismatch_max(nao_by_count, monomer["nao"]),
        "all_finite": bool(finite),
        "cosx_monomer_max_abs": monomer["cosx_max_abs"],
        "cosx_full_tensor_calls": int(monomer["cosx_full_tensor_calls"]) + sum(
            int(clusters[str(count)].get("cosx_full_tensor_calls", 0))
            for count in CLUSTER_COUNTS
        ),
        "jk_16_s": clusters["16"]["jk_seconds"],
        "rss_16_gib": clusters["16"]["rss_gib"],
    }
    metrics.update(errors)
    for count in CLUSTER_COUNTS:
        cluster = clusters[str(count)]
        metrics[f"nao_{count}"] = cluster["nao"]
        metrics[f"significant_pair_fraction_{count}"] = cluster["significant_pair_fraction"]
        metrics[f"memory_{count}_gib"] = cluster["estimated_dense_working_set_gib"]
        metrics[f"skipped_{count}"] = bool(cluster["skipped"])
        if "jk_seconds" in cluster:
            metrics[f"jk_{count}_s"] = cluster["jk_seconds"]
        if "estimated_seconds" in cluster:
            metrics[f"estimated_seconds_{count}"] = cluster["estimated_seconds"]
    report = {
        "metrics": metrics,
        "details": {
            "basis": BASIS,
            "auxbasis": AUXBASIS,
            "df_tol": DF_TOL,
            "grid_rad": GRID_RAD,
            "grid_ang": GRID_ANG,
            "batch_size": BATCH_SIZE,
            "jk_scf_executed": False,
            "monomer": monomer,
            "clusters": clusters,
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
