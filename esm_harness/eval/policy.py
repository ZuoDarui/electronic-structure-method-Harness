"""Static rejection rules for generated programs."""

from __future__ import annotations

import re

_SCF_IMPORT = re.compile(r"pyscf\s*\.\s*scf|\bimport\s+scf\b|from\s+pyscf\s+import\s+[^#\n]*\bscf\b")
_DF_IMPORT = re.compile(r"pyscf\s*\.\s*df|from\s+pyscf\s+import\s+[^#\n]*\bdf\b")
_SCF_DRIVER = re.compile(r"\bscf\s*\.\s*(RHF|UHF|ROHF|GHF|RKS|UKS)\b")
_KERNEL_CALL = re.compile(r"\.kernel\s*\(")


def rejection_reason(source: str, allowed_imports: list[str]) -> str | None:
    """Return why ``source`` cannot be evaluated, or None when it is allowed."""
    allowed = set(allowed_imports)
    if "pyscf.scf" not in allowed and (
        _SCF_IMPORT.search(source) or _SCF_DRIVER.search(source)
    ):
        return "pyscf.scf drivers are not allowed in generated programs"
    if "pyscf.df" not in allowed and _DF_IMPORT.search(source):
        return "pyscf.df is not allowed in generated programs"
    if _KERNEL_CALL.search(source):
        return "calling kernel() is not allowed; expose solve(mol)"
    if "def solve" not in source:
        return "generated program must define solve(mol)"
    return None


def blocked_modules(allowed_imports: list[str]) -> list[str]:
    blocked = []
    allowed = set(allowed_imports)
    if "pyscf.scf" not in allowed:
        blocked.append("pyscf.scf")
    if "pyscf.df" not in allowed:
        blocked.append("pyscf.df")
    return blocked
