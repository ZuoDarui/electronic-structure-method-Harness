"""PySCF reference energies and the in-process baseline accuracy gate."""

from __future__ import annotations

import json
from pathlib import Path

from esm_harness.eval.cases import load_cases
from esm_harness.eval.worker import _load_solve


def build_mol(case: dict):
    from pyscf import gto

    return gto.M(
        atom=case["atom"],
        basis=case["basis"],
        unit=case["unit"],
        charge=case["charge"],
        spin=case["spin"],
        verbose=0,
    )


def pyscf_energies(cases: list[dict]) -> dict[str, float]:
    from pyscf import scf

    energies = {}
    for case in cases:
        mol = build_mol(case)
        mean_field = scf.RHF(mol)
        mean_field.conv_tol = 1e-12
        energy = float(mean_field.kernel())
        if not mean_field.converged:
            raise RuntimeError(f"PySCF RHF did not converge for {case['id']}")
        energies[case["id"]] = energy
    return energies


def write_references(cases: list[dict], path: Path) -> dict[str, float]:
    energies = pyscf_energies(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(energies, indent=2) + "\n", encoding="utf-8")
    return energies


def read_references(path: Path, cases: list[dict]) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [case["id"] for case in cases if case["id"] not in data]
    if missing:
        raise ValueError(f"{path} is missing reference energies for {missing}")
    return {case["id"]: float(data[case["id"]]) for case in cases}


def baseline_errors(baseline: Path, cases: list[dict], references: dict[str, float]) -> list[dict]:
    solve = _load_solve(baseline)
    rows = []
    for case in cases:
        result = solve(build_mol(case))
        energy = float(result["e_tot"])
        rows.append(
            {
                "id": case["id"],
                "energy": energy,
                "ref_energy": references[case["id"]],
                "abs_err": abs(energy - references[case["id"]]),
                "converged": bool(result["converged"]),
            }
        )
    return rows
