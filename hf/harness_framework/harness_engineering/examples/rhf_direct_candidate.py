"""Reference-shaped candidate for the direct RHF harness task.

It intentionally emits only a small JSON report. The harness owns the decision;
the candidate owns the scientific calculation and raw evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from pyscf import gto, scf

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
from hf import Hartree_Fock  # noqa: E402


CASES = [
    ("h2-sto3g", "H 0 0 0; H 0 0 1", "sto-3g"),
    ("h2-631g", "H 0 0 0; H 0 0 1", "6-31g"),
    ("water-sto3g", "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "sto-3g"),
]


def run_case(atom: str, basis: str) -> tuple[float, float, float, bool]:
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    dm = scf.RHF(mol).get_init_guess()
    solver = Hartree_Fock(mol)
    direct_j, direct_k = solver.build_jk(dm)
    ref_j, ref_k = scf.RHF(mol).get_jk(mol, dm)
    jk_error = max(float(np.max(np.abs(direct_j - ref_j))), float(np.max(np.abs(direct_k - ref_k))))
    energy, _ = solver.solve_rhf(mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-9)
    ref = scf.RHF(mol)
    ref.conv_tol = 1e-12
    ref.kernel()
    return abs(float(energy - ref.e_tot)), jk_error, float(energy), bool(ref.converged)


def main() -> int:
    energy_errors, jk_errors, energies, converged = [], [], [], []
    timings = []
    for _, atom, basis in CASES:
        start = time.perf_counter()
        energy_error, jk_error, energy, ref_converged = run_case(atom, basis)
        timings.append(time.perf_counter() - start)
        energy_errors.append(energy_error)
        jk_errors.append(jk_error)
        energies.append(energy)
        converged.append(ref_converged)
    report = {
        "metrics": {
            "energy_max_abs": max(energy_errors),
            "jk_max_abs": max(jk_errors),
            "all_finite": bool(np.isfinite(energies).all()),
            "scf_converged": all(converged),
            "median_seconds": statistics.median(timings),
            "case_count": len(CASES),
        },
        "cases": [name for name, _, _ in CASES],
    }
    report_path = os.environ.get("HARNESS_REPORT_PATH")
    if not report_path:
        print(json.dumps(report, indent=2))
    else:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
