"""Run configuration, ablation grids, and repository paths."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

FACTOR_KEYS = (
    "model",
    "prompt_id",
    "search",
    "speed_weight",
    "reasoning_effort",
    "population_n",
    "allowed_imports",
    "max_abs_err",
    "max_time_ratio",
    "fitness_err_ceiling",
)

SEARCHES = {"iterative", "best_of_n"}
PROMPTS = {"hf_minimal", "hf_spec", "hf_show_baseline"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# Cursor subagent slugs encode the reasoning effort in the model name.
SUBAGENT_MODELS = {"claude-opus-5.5": "claude-opus-5-5"}


def subagent_model(factors: dict) -> str:
    model = str(factors["model"])
    try:
        stem = SUBAGENT_MODELS[model]
    except KeyError as exc:
        raise ValueError(f"no subagent slug for model {model!r}") from exc
    return f"{stem}-{factors['reasoning_effort']}"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def validate_factors(factors: dict) -> dict:
    missing = [key for key in FACTOR_KEYS if key not in factors]
    if missing:
        raise ValueError(f"run factors missing {missing}")
    if factors["search"] not in SEARCHES:
        raise ValueError(f"unknown search {factors['search']!r}")
    if factors["prompt_id"] not in PROMPTS:
        raise ValueError(f"unknown prompt_id {factors['prompt_id']!r}")
    if factors["reasoning_effort"] not in EFFORTS:
        raise ValueError(f"unknown reasoning_effort {factors['reasoning_effort']!r}")
    if int(factors["population_n"]) < 1:
        raise ValueError("population_n must be >= 1")
    normalized = dict(factors)
    normalized["population_n"] = int(factors["population_n"])
    normalized["speed_weight"] = float(factors["speed_weight"])
    normalized["max_abs_err"] = float(factors["max_abs_err"])
    normalized["max_time_ratio"] = float(factors["max_time_ratio"])
    normalized["fitness_err_ceiling"] = float(factors["fitness_err_ceiling"])
    normalized["allowed_imports"] = list(factors["allowed_imports"])
    return normalized


def _cell(base: dict, overrides: dict) -> dict:
    factors = dict(base["factors"])
    factors.update(overrides)
    cell = dict(base)
    cell["factors"] = validate_factors(factors)
    return cell


def cell_id(index: int, factors: dict) -> str:
    return (
        f"cell-{index:02d}-{factors['prompt_id']}-{factors['search']}-"
        f"w{factors['speed_weight']}-{factors['reasoning_effort']}"
    )


def expand_grid(document: dict) -> list[dict]:
    """Return one run config per ablation cell.

    ``grid`` is a full factorial over the listed factors. ``cells`` is an
    explicit list of factor overrides, for one-factor-at-a-time designs.
    """
    base = document["base"]
    grid = document.get("grid") or {}
    explicit = document.get("cells") or []
    if grid and explicit:
        raise ValueError("use either grid or cells, not both")
    if explicit:
        return [_cell(base, dict(overrides or {})) for overrides in explicit]
    keys = list(grid)
    if not keys:
        return [_cell(base, {})]
    return [
        _cell(base, dict(zip(keys, combo)))
        for combo in itertools.product(*(grid[key] for key in keys))
    ]
