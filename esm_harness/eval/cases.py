"""Fixed evaluation suites."""

from __future__ import annotations

from pathlib import Path

from esm_harness.config import load_yaml


def load_cases(path: str | Path) -> list[dict]:
    document = load_yaml(path)
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} has no cases")
    normalized = []
    seen = set()
    for case in cases:
        for key in ("id", "atom", "basis"):
            if key not in case:
                raise ValueError(f"case missing {key}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
        normalized.append(
            {
                "id": str(case["id"]),
                "atom": str(case["atom"]).strip(),
                "basis": str(case["basis"]),
                "charge": int(case.get("charge", 0)),
                "spin": int(case.get("spin", 0)),
                "unit": str(case.get("unit", "Angstrom")),
            }
        )
    return normalized
