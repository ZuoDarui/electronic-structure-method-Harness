"""Compare the replaceable density update with the dense reference.

The matrices are the core Hamiltonian and overlap of one water monomer and
the smallest one-dimensional water, benzene, and alkane cases. This task does
not time the ladder and does not modify hf.py.
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
from harness_engineering.examples.linear_get_jk_candidate import (  # noqa: E402
    BASIS,
    BENZENE,
    BENZENE_SPACING,
    WATER_MONOMER,
    WATER_SPACING,
    alkane_atom_string,
    translated,
)
from harness_engineering.generated.occupy_density import occupy_density  # noqa: E402
from harness_engineering.training_hf import (  # noqa: E402
    density_asymmetry,
    density_electrons,
    occupy_density_reference,
)


def run_case(name, atom):
    mol = gto.M(atom=atom, basis=BASIS, unit="Angstrom", verbose=0)
    n_alpha, n_beta = mol.nelec
    if n_alpha != n_beta:
        raise ValueError(f"{name} is not a closed shell")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    reference = np.asarray(
        occupy_density_reference(hcore, overlap, n_alpha), dtype=np.float64)
    candidate = np.asarray(occupy_density(hcore, overlap, n_alpha), dtype=np.float64)
    electrons = density_electrons(overlap, candidate)
    finite = bool(
        np.isfinite(reference).all()
        and np.isfinite(candidate).all()
        and np.isfinite(hcore).all()
        and np.isfinite(overlap).all()
    )
    return {
        "name": name,
        "nao": int(mol.nao_nr()),
        "density_max_abs": float(np.max(np.abs(candidate - reference))),
        "density_asymmetry": density_asymmetry(candidate),
        "density_electron_abs": abs(float(electrons) - float(mol.nelectron)),
        "all_finite": finite,
    }


def main():
    lib.num_threads(1)
    cases = []
    for name, builder in (
        ("water-monomer", lambda: translated(WATER_MONOMER, 1, 0, WATER_SPACING)),
        ("water-8", lambda: translated(WATER_MONOMER, 8, 2, WATER_SPACING)),
        ("benzene-2", lambda: translated(BENZENE, 2, 0, BENZENE_SPACING)),
        ("alkane-8", lambda: alkane_atom_string(8)),
    ):
        record = run_case(name, builder())
        print(
            f"{name} nao={record['nao']} density={record['density_max_abs']:.3e} "
            f"electrons={record['density_electron_abs']:.3e}",
            file=sys.stderr,
            flush=True,
        )
        cases.append(record)
    finite = all(record["all_finite"] for record in cases)
    report = {
        "metrics": {
            "all_finite": bool(finite),
            "density_max_abs": max(record["density_max_abs"] for record in cases),
            "density_asymmetry_max": max(record["density_asymmetry"] for record in cases),
            "density_electron_abs_max": max(
                record["density_electron_abs"] for record in cases),
        },
        "details": {
            "basis": BASIS,
            "fock": "core Hamiltonian",
            "reference": "harness_engineering.training_hf.occupy_density_reference",
            "candidate": "harness_engineering.generated.occupy_density",
            "cases": cases,
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
