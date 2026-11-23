"""Versioned, explicit scientific campaign specifications.

These specifications describe a known method to implement, not a license to
change its equations, reference answers or validation thresholds while tuning.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA_VERSION = 1
CASE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}$")
DEFAULT_TOLERANCES = {
    "energy_max_abs": 1e-8,
    "jk_max_abs": 1e-9,
    "jk_symmetry_max_abs": 1e-11,
    "jk_linearity_max_abs": 1e-9,
    "density_max_abs": 1e-6,
    "density_symmetry_max_abs": 1e-11,
    "electron_count_abs": 1e-9,
    "idempotency_max_abs": 1e-8,
    "residual_max_abs": 1e-7,
    "orthogonality_max_abs": 1e-10,
    "orbital_equation_max_abs": 1e-7,
    "density_reconstruction_max_abs": 1e-10,
    "energy_reconstruction_abs": 1e-9,
    "occupation_max_abs": 1e-12,
    "orbital_order_error": 1e-12,
    "anchor_energy_abs": 1e-9,
}


def validate_spec(raw: dict) -> dict:
    """Return an independent validated copy; never silently omit a case."""
    if (not isinstance(raw, dict) or isinstance(raw.get("schema_version"), bool)
            or raw.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("campaign schema_version must be 1")
    spec = copy.deepcopy(raw)
    if not isinstance(spec.get("name"), str) or not CASE_ID.fullmatch(spec["name"]):
        raise ValueError("campaign name must be a safe identifier")
    if spec.get("method") != "RHF":
        raise ValueError("first generation protocol supports RHF only")
    modes = spec.get("integral_modes")
    if (not isinstance(modes, list) or not modes or any(not isinstance(mode, str) for mode in modes)
            or len(set(modes)) != len(modes)
            or any(mode not in {"exact", "df"} for mode in modes)):
        raise ValueError("integral_modes must contain distinct exact/df values")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    names = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        name = case.get("id", case.get("name"))
        if not isinstance(name, str) or not CASE_ID.fullmatch(name):
            raise ValueError("every case needs a safe id or name")
        case["id"] = name
        if not isinstance(case.get("atom"), (str, list)) or not case["atom"]:
            raise ValueError(f"{name}: atom geometry is required")
        if not isinstance(case.get("basis"), (str, dict)) or not case["basis"]:
            raise ValueError(f"{name}: basis is required")
        if case.get("spin", 0) != 0 or isinstance(case.get("spin", 0), bool):
            raise ValueError(f"{name}: RHF requires spin=0")
        if not isinstance(case.get("charge", 0), int) or isinstance(case.get("charge", 0), bool):
            raise ValueError(f"{name}: charge must be an integer")
        if case.get("unit", "Angstrom") not in {"Angstrom", "Bohr"}:
            raise ValueError(f"{name}: unit must be Angstrom or Bohr")
        if "anchor_energy" in case:
            anchor = case["anchor_energy"]
            if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or not math.isfinite(anchor):
                raise ValueError(f"{name}: anchor_energy must be finite and numerical")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("duplicate case ids")
    if "df" in modes and not isinstance(spec.get("auxbasis"), str):
        raise ValueError("df requires an explicit auxbasis")
    tolerances = spec.get("tolerances")
    if not isinstance(tolerances, dict) or set(tolerances) != set(DEFAULT_TOLERANCES):
        raise ValueError("tolerances must explicitly cover the complete RHF metric protocol")
    for key, value in tolerances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid tolerance: {key}")
    memory = spec.get("memory_limit_mb", 2048)
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError("memory_limit_mb must be finite and positive")
    seed = spec.get("seed", 20261111)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    spec["memory_limit_mb"], spec["seed"] = memory, seed
    solver = spec.get("solver", {})
    if not isinstance(solver, dict):
        raise ValueError("solver must be an object")
    for key in ("conv_tol", "residual_tol"):
        value = solver.get(key, 1e-10 if key == "conv_tol" else 1e-8)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"solver.{key} must be finite and positive")
    for key, minimum in (("max_cycle", 1), ("diis_space", 0), ("df_block_size", 1)):
        value = solver.get(key, {"max_cycle": 150, "diis_space": 8, "df_block_size": 32}[key])
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"solver.{key} must be an integer >= {minimum}")
    generation = spec.get("generation")
    if generation is not None:
        if not isinstance(generation, dict):
            raise ValueError("generation must be an object")
        rounds = generation.get("max_rounds", 8)
        if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 8:
            raise ValueError("generation.max_rounds must be an integer from 1 to 8")
        if generation.get("provider", "openai_responses") != "openai_responses":
            raise ValueError("generation.provider must be openai_responses")
        if generation.get("candidate_file", "solution.py") != "solution.py":
            raise ValueError("generation.candidate_file must be solution.py")
    execution = spec.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    for key in ("timeout_s", "memory_limit_mb"):
        value = execution.get(key, 300 if key == "timeout_s" else memory)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"execution.{key} must be finite and positive")
    for key in ("require_external_trace", "allow_gpu_fallback"):
        if key in execution and not isinstance(execution[key], bool):
            raise ValueError(f"execution.{key} must be boolean")
    return spec


def load_spec(path: str | Path) -> dict:
    return validate_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def expand_cases(spec: dict) -> list[dict]:
    """Expand six geometries × two modes into twelve required executions."""
    spec = validate_spec(spec)
    expanded = []
    for case in spec["cases"]:
        for mode in spec["integral_modes"]:
            record = copy.deepcopy(case)
            record["base_id"] = record["id"]
            record["id"] += "-" + mode
            if not CASE_ID.fullmatch(record["id"]):
                raise ValueError("expanded case id exceeds the identifier length limit")
            record["mode"] = mode
            record["auxbasis"] = spec.get("auxbasis", "weigend")
            expanded.append(record)
    if len({case["id"] for case in expanded}) != len(expanded):
        raise ValueError("expanded case ids collide")
    return expanded


def spec_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(validate_spec(spec), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
