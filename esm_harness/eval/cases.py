"""Fixed evaluation suites."""

from __future__ import annotations

from pathlib import Path

from esm_harness.config import load_yaml


def read_xyz(path: Path) -> str:
    """PySCF atom string from an XYZ file, read the way the cluster's hf.py reads it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    natm = int(lines[0].strip())
    atoms = [" ".join(line.split()[:4]) for line in lines[2:] if line.strip()]
    if len(atoms) != natm:
        raise ValueError(f"{path}: header says {natm} atoms, found {len(atoms)}")
    return ";".join(atoms)


def load_cases(path: str | Path) -> list[dict]:
    path = Path(path)
    document = load_yaml(path)
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} has no cases")
    suite_method = document.get("method")
    normalized = []
    seen = set()
    for case in cases:
        for key in ("id", "basis"):
            if key not in case:
                raise ValueError(f"case missing {key}")
        if ("atom" in case) == ("xyz" in case):
            raise ValueError(f"case {case['id']} needs exactly one of atom or xyz")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
        atom = str(case["atom"]).strip() if "atom" in case else read_xyz(path.parent / case["xyz"])
        item = {
            "id": str(case["id"]),
            "atom": atom,
            "basis": str(case["basis"]),
            "charge": int(case.get("charge", 0)),
            "spin": int(case.get("spin", 0)),
            "unit": str(case.get("unit", "Angstrom")),
        }
        if "xyz" in case:
            item["xyz"] = str(case["xyz"])
        if "nao" in case:
            item["nao"] = int(case["nao"])
        if suite_method or case.get("method"):
            item["method"] = {**(suite_method or {}), **(case.get("method") or {})}
        normalized.append(item)
    return normalized
