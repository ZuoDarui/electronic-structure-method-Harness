"""Reference energies and the in-process baseline accuracy gate.

A references file maps each case id either to an energy, or to the energies
that several programs recorded for it::

    {"water8": {"pyscf": {"energy": ..., "converged": true, "time_s": ...},
                "hf.py": {"energy": ..., "converged": true, "time_s": ...}}}

In the second form the reference is the first converged entry in
``REFERENCE_ORDER``: PySCF when it has a result, otherwise hf.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from esm_harness.eval.worker import _build_mol, _call_solve, _load_solve

REFERENCE_ORDER = ("pyscf", "hf.py")


def build_mol(case: dict):
    return _build_mol(case)


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


def select_reference(case_id: str, entry) -> dict:
    """Return ``{"energy", "source", "time_s"}`` for one case's references entry."""
    if isinstance(entry, (int, float)):
        return {"energy": float(entry), "source": None, "time_s": None}
    for source in REFERENCE_ORDER:
        record = entry.get(source) or {}
        if record.get("converged") and record.get("energy") is not None:
            time_s = record.get("time_s")
            return {
                "energy": float(record["energy"]),
                "source": source,
                "time_s": None if time_s is None else float(time_s),
            }
    raise ValueError(f"{case_id} has no converged reference from any of {REFERENCE_ORDER}")


def read_reference_records(path: Path, cases: list[dict]) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [case["id"] for case in cases if case["id"] not in data]
    if missing:
        raise ValueError(f"{path} is missing reference energies for {missing}")
    return {case["id"]: select_reference(case["id"], data[case["id"]]) for case in cases}


def read_references(path: Path, cases: list[dict]) -> dict[str, float]:
    return {case_id: record["energy"] for case_id, record in read_reference_records(path, cases).items()}


def baseline_errors(baseline: Path, cases: list[dict], references: dict[str, float]) -> list[dict]:
    solve = _load_solve(baseline)
    rows = []
    for case in cases:
        result = _call_solve(solve, case)
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
