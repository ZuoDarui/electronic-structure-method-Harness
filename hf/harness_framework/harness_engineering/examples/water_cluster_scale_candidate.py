"""Record scale, memory, and one-electron diagonalization for water clusters.

This candidate builds the reference Hartree_Fock object but does not call
J/K or the SCF loop. Those costs belong to the next optimization stage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from pyscf import gto, lib
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
from hf import Hartree_Fock, estimate_dense_solver_memory_gib  # noqa: E402
from harness_engineering.benchmark import benchmark  # noqa: E402
from harness_engineering.water_clusters import (  # noqa: E402
    CLUSTER_COUNTS,
    cluster_atom_string,
    nao_mismatch_max,
)


BASIS = "cc-pvtz"
TIMING = {
    16: (1, 3),
    32: (1, 3),
    64: (0, 1),
    128: (0, 1),
}
MONOMER_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"


def _timed(function, count):
    warmups, repeats = TIMING[count]
    holder = {}

    def wrapped():
        holder["value"] = function()

    summary = benchmark(wrapped, warmups=warmups, repeats=repeats)
    payload = summary.to_dict()
    payload["warmups"] = warmups
    payload["repeats"] = repeats
    return payload, holder["value"]


def run_cluster(count, monomer_nao):
    mol = gto.M(
        atom=cluster_atom_string(count),
        basis=BASIS,
        unit="Angstrom",
        verbose=0,
    )
    reference = Hartree_Fock(mol)
    nao = int(mol.nao_nr())
    memory = estimate_dense_solver_memory_gib(nao)

    def core_hamiltonian():
        return mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    def overlap_matrix():
        return mol.intor("int1e_ovlp")

    hcore_timing, hcore = _timed(core_hamiltonian, count)
    overlap_timing, overlap = _timed(overlap_matrix, count)

    def diagonalize():
        return eigh(hcore, overlap)

    eigh_timing, diagonalized = _timed(diagonalize, count)
    orbital_energy, coefficients = diagonalized
    finite = bool(
        np.isfinite(hcore).all()
        and np.isfinite(overlap).all()
        and np.isfinite(orbital_energy).all()
        and np.isfinite(coefficients).all()
    )
    return {
        "molecules": count,
        "basis": BASIS,
        "natm": int(mol.natm),
        "nao": nao,
        "expected_nao": int(monomer_nao) * count,
        "nshells": int(reference.nshells),
        "nelectron": int(mol.nelectron),
        "parallel_active": bool(reference.parallel_active),
        "estimated_dense_working_set_gib": memory["estimated_dense_working_set_gib"],
        "one_dense_matrix_gib": memory["one_dense_matrix_gib"],
        "all_finite": finite,
        "hcore_timing": hcore_timing,
        "overlap_timing": overlap_timing,
        "eigh_timing": eigh_timing,
    }


def summarize(monomer_nao, clusters):
    nao_by_count = {count: clusters[str(count)]["nao"] for count in CLUSTER_COUNTS}
    finite = all(clusters[str(count)]["all_finite"] for count in CLUSTER_COUNTS)
    metrics = {
        "monomer_nao": int(monomer_nao),
        "nao_mismatch_max": nao_mismatch_max(nao_by_count, monomer_nao),
        "all_finite": finite,
    }
    for count in CLUSTER_COUNTS:
        cluster = clusters[str(count)]
        metrics[f"nao_{count}"] = cluster["nao"]
        metrics[f"memory_{count}_gib"] = cluster["estimated_dense_working_set_gib"]
        metrics[f"hcore_{count}_median_s"] = cluster["hcore_timing"]["median_s"]
        metrics[f"overlap_{count}_median_s"] = cluster["overlap_timing"]["median_s"]
        metrics[f"eigh_{count}_median_s"] = cluster["eigh_timing"]["median_s"]
    return metrics


def main():
    lib.num_threads(1)
    monomer = gto.M(atom=MONOMER_ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    monomer_nao = int(monomer.nao_nr())
    clusters = {}
    for count in CLUSTER_COUNTS:
        clusters[str(count)] = run_cluster(count, monomer_nao)
    report = {
        "metrics": summarize(monomer_nao, clusters),
        "details": {
            "basis": BASIS,
            "long_term_target": {
                "basis_functions": 50000,
                "scf_iterations": 20,
                "convergence_hartree": 1e-8,
                "wall_time_hours": 48,
            },
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
