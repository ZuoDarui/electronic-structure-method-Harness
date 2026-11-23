"""Time the MINAO density guess and one generalized diagonalization.

The diagonalization uses the core Hamiltonian and the overlap, which is the
same dense kernel the SCF loop calls. This stage does not construct
Hartree_Fock and does not run J/K or the SCF loop.
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
from pyscf import gto, lib, scf
from scipy.linalg import eigh

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
from hf import estimate_dense_solver_memory_gib  # noqa: E402
from harness_engineering.water_clusters import (  # noqa: E402
    CLUSTER_COUNTS,
    cluster_atom_string,
    nao_mismatch_max,
)


BASIS = "cc-pvtz"
TASK_TIMEOUT_S = 3600
REPORT_RESERVE_S = 180
MONOMER_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"


def estimate_eigh_seconds(previous_seconds, previous_nao, next_nao):
    """Scale a finished diagonalization by the cube of the basis-set size."""

    if (previous_seconds is None or previous_seconds <= 0
            or previous_nao <= 0 or next_nao <= 0):
        return None
    ratio = float(next_nao) / float(previous_nao)
    return float(previous_seconds) * ratio ** 3


def should_skip_eigh(estimated_seconds, remaining_seconds):
    return estimated_seconds is not None and estimated_seconds > remaining_seconds


def peak_rss_gib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)
    return rss / (1024 ** 2) / 1024


def _gib(array):
    return float(np.asarray(array).nbytes) / (1024 ** 3)


def _asymmetry(matrix):
    return float(np.max(np.abs(matrix - matrix.T)))


def _electron_count(overlap, density):
    return float(np.einsum("ij,ji->", overlap, density))


def generalized_residual(hcore, overlap, energy, coeff):
    """Largest column residual of H C - S C e, scaled by max(1, |e|)."""

    energy = np.asarray(energy)
    coeff = np.asarray(coeff)
    residual = hcore @ coeff - (overlap @ coeff) * energy
    scale = np.maximum(1.0, np.abs(energy))
    return float(np.max(np.linalg.norm(residual, axis=0) / scale))


def _remaining_seconds(started):
    elapsed = time.perf_counter() - started
    return TASK_TIMEOUT_S - elapsed - REPORT_RESERVE_S


def measure_cluster(count, monomer_nao, started, previous):
    mol = gto.M(
        atom=cluster_atom_string(count),
        basis=BASIS,
        unit="Angstrom",
        verbose=0,
    )
    nao = int(mol.nao_nr())
    nelectron = int(mol.nelectron)
    memory = estimate_dense_solver_memory_gib(nao)
    guess_started = time.perf_counter()
    density = np.asarray(scf.hf.init_guess_by_minao(mol))
    guess_seconds = time.perf_counter() - guess_started
    overlap_started = time.perf_counter()
    overlap = mol.intor("int1e_ovlp")
    overlap_seconds = time.perf_counter() - overlap_started
    electrons = _electron_count(overlap, density)
    record = {
        "molecules": count,
        "basis": BASIS,
        "natm": int(mol.natm),
        "nao": nao,
        "expected_nao": int(monomer_nao) * count,
        "nelectron": nelectron,
        "estimated_dense_working_set_gib": memory["estimated_dense_working_set_gib"],
        "one_dense_matrix_gib": memory["one_dense_matrix_gib"],
        "guess_seconds": guess_seconds,
        "guess_output_gib": _gib(density),
        "guess_asymmetry": _asymmetry(density),
        "guess_electron_abs": abs(electrons - nelectron),
        "guess_electron_rel": abs(electrons - nelectron) / float(nelectron),
        "overlap_seconds": overlap_seconds,
        "rss_guess_gib": peak_rss_gib(),
        "guess_finite": bool(np.isfinite(density).all() and np.isfinite(overlap).all()),
        "eigh_skipped": False,
    }
    print(
        f"guess molecules={count} nao={nao} seconds={guess_seconds:.3f} "
        f"output_gib={record['guess_output_gib']:.6f} "
        f"rss_gib={record['rss_guess_gib']:.3f}",
        file=sys.stderr,
        flush=True,
    )
    remaining = _remaining_seconds(started)
    estimate = None
    if previous is not None:
        estimate = estimate_eigh_seconds(
            previous["seconds"], previous["nao"], nao)
    if count != 16 and should_skip_eigh(estimate, remaining):
        record["eigh_skipped"] = True
        record["estimated_eigh_seconds"] = estimate
        record["skip_reason"] = (
            f"estimated diagonalization time {estimate:.3f} s exceeds "
            f"the remaining {remaining:.3f} s"
        )
        print(record["skip_reason"], file=sys.stderr, flush=True)
        del density, overlap
        return record, previous

    hcore_started = time.perf_counter()
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    record["hcore_seconds"] = time.perf_counter() - hcore_started
    if count == 16:
        eigh(hcore, overlap)
    eigh_started = time.perf_counter()
    energy, coeff = eigh(hcore, overlap)
    eigh_seconds = time.perf_counter() - eigh_started
    nocc = nelectron // 2
    occupied = np.asarray(coeff[:, :nocc])
    built = 2.0 * (occupied @ occupied.T)
    record.update({
        "eigh_seconds": eigh_seconds,
        "eigh_output_gib": _gib(energy) + _gib(coeff),
        "rss_eigh_gib": peak_rss_gib(),
        "eigh_finite": bool(np.isfinite(energy).all() and np.isfinite(coeff).all()
                            and np.isfinite(hcore).all() and np.isfinite(built).all()),
        "density_trace_abs": abs(_electron_count(overlap, built) - nelectron),
        "density_asymmetry": _asymmetry(built),
    })
    if count == 16:
        record["eigh_residual"] = generalized_residual(hcore, overlap, energy, coeff)
    print(
        f"eigh molecules={count} nao={nao} seconds={eigh_seconds:.3f} "
        f"output_gib={record['eigh_output_gib']:.6f} "
        f"rss_gib={record['rss_eigh_gib']:.3f} remaining={remaining:.1f}",
        file=sys.stderr,
        flush=True,
    )
    del density, overlap, hcore, energy, coeff, occupied, built
    return record, {"seconds": eigh_seconds, "nao": nao}


def main():
    lib.num_threads(1)
    started = time.perf_counter()
    monomer = gto.M(atom=MONOMER_ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    monomer_nao = int(monomer.nao_nr())
    clusters = {}
    previous = None
    for count in CLUSTER_COUNTS:
        record, previous = measure_cluster(count, monomer_nao, started, previous)
        clusters[str(count)] = record

    nao_by_count = {count: clusters[str(count)]["nao"] for count in CLUSTER_COUNTS}
    finite = all(
        clusters[str(count)]["guess_finite"]
        and clusters[str(count)].get("eigh_finite", True)
        for count in CLUSTER_COUNTS
    )
    metrics = {
        "monomer_nao": monomer_nao,
        "nao_mismatch_max": nao_mismatch_max(nao_by_count, monomer_nao),
        "all_finite": bool(finite),
        "guess_asymmetry_16": clusters["16"]["guess_asymmetry"],
        "guess_electron_rel_16": clusters["16"]["guess_electron_rel"],
        "eigh_16_s": clusters["16"]["eigh_seconds"],
        "eigh_residual_16": clusters["16"]["eigh_residual"],
        "density_trace_16_abs": clusters["16"]["density_trace_abs"],
        "density_asymmetry_16": clusters["16"]["density_asymmetry"],
        "rss_guess_16_gib": clusters["16"]["rss_guess_gib"],
        "rss_eigh_16_gib": clusters["16"]["rss_eigh_gib"],
    }
    for count in CLUSTER_COUNTS:
        cluster = clusters[str(count)]
        metrics[f"nao_{count}"] = cluster["nao"]
        metrics[f"memory_{count}_gib"] = cluster["estimated_dense_working_set_gib"]
        metrics[f"guess_{count}_s"] = cluster["guess_seconds"]
        metrics[f"guess_output_{count}_gib"] = cluster["guess_output_gib"]
        metrics[f"guess_electron_rel_{count}"] = cluster["guess_electron_rel"]
        metrics[f"skipped_eigh_{count}"] = bool(cluster["eigh_skipped"])
        if "eigh_seconds" in cluster:
            metrics[f"eigh_{count}_s"] = cluster["eigh_seconds"]
            metrics[f"eigh_output_{count}_gib"] = cluster["eigh_output_gib"]
            metrics[f"rss_eigh_{count}_gib"] = cluster["rss_eigh_gib"]
            metrics[f"density_trace_abs_{count}"] = cluster["density_trace_abs"]
        if "estimated_eigh_seconds" in cluster:
            metrics[f"estimated_eigh_seconds_{count}"] = cluster["estimated_eigh_seconds"]
    report = {
        "metrics": metrics,
        "details": {
            "basis": BASIS,
            "guess": "minao",
            "diagonalized_matrix": "hcore",
            "threads": 1,
            "jk_scf_executed": False,
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
