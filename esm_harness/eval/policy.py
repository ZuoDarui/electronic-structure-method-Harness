"""Static rejection rules for generated programs."""

from __future__ import annotations

import ast
import re

_SCF_IMPORT = re.compile(r"pyscf\s*\.\s*scf|\bimport\s+scf\b|from\s+pyscf\s+import\s+[^#\n]*\bscf\b")
_DF_IMPORT = re.compile(r"pyscf\s*\.\s*df|from\s+pyscf\s+import\s+[^#\n]*\bdf\b")
_SCF_DRIVER = re.compile(r"\bscf\s*\.\s*(RHF|UHF|ROHF|GHF|RKS|UKS)\b")
_KERNEL_CALL = re.compile(r"\.kernel\s*\(")


# Pure helpers with no file, process, or network access.
SAFE_STDLIB = {"math", "cmath", "itertools", "functools", "collections", "typing", "dataclasses", "__future__"}

# Persisting results between timed calls would let a program skip work.
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "breakpoint", "input"}
_FORBIDDEN_ATTRS = {
    "save", "savez", "savez_compressed", "load", "memmap", "tofile", "fromfile",
    "loadtxt", "savetxt", "genfromtxt", "fromregex",
}


def _module_allowed(name: str, allowed: set[str]) -> bool:
    if name.split(".")[0] in SAFE_STDLIB:
        return True
    return any(name == root or name.startswith(root + ".") for root in allowed)


def _ast_reason(source: str, allowed: set[str]) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "relative imports are not allowed"
            base = node.module or ""
            names = [base] if base != "pyscf" else [f"pyscf.{alias.name}" for alias in node.names]
        else:
            names = []
        for name in names:
            if not _module_allowed(name, allowed):
                return f"import of {name} is not allowed; allowed: {sorted(allowed)}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                return f"{func.id}() is not allowed in generated programs"
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_ATTRS:
                return f".{func.attr}() file I/O is not allowed in generated programs"
    return None


def rejection_reason(source: str, allowed_imports: list[str]) -> str | None:
    """Return why ``source`` cannot be evaluated, or None when it is allowed."""
    allowed = set(allowed_imports)
    reason = _ast_reason(source, allowed)
    if reason:
        return reason
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
