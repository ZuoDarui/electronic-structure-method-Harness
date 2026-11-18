"""JSONL trajectories and token ledgers for later ablation."""

from __future__ import annotations

import json
from pathlib import Path

USAGE_FIELDS = (
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def empty_usage() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS}


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _add_usage(total: dict[str, int], usage: dict | None) -> None:
    if not usage:
        return
    for field in USAGE_FIELDS:
        total[field] += int(usage.get(field) or 0)


def best_record(records: list[dict]) -> dict | None:
    """Fittest record within the acceptance floor; records scored before the floor
    existed may carry a fitness with a larger error, and those never qualify."""
    from esm_harness.eval.score import ACCEPTANCE_FLOOR

    ranked = [
        record
        for record in records
        if record["eval"].get("fitness") is not None
        and record["eval"].get("max_abs_err") is not None
        and float(record["eval"]["max_abs_err"]) <= ACCEPTANCE_FLOOR
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda record: float(record["eval"]["fitness"]))


def best_summary(record: dict | None) -> dict | None:
    if record is None:
        return None
    evaluation = record["eval"]
    return {
        "candidate_id": record["candidate_id"],
        "generation": record["generation"],
        "fitness": evaluation["fitness"],
        "max_abs_err": evaluation["max_abs_err"],
        "max_abs_err_vs_baseline": evaluation.get("max_abs_err_vs_baseline"),
        "geomean_time_ratio": evaluation["geomean_time_ratio"],
        "status": evaluation["status"],
        "cases": [
            {
                "id": case["id"],
                "abs_err": case["abs_err"],
                "abs_err_baseline": case["abs_err_baseline"],
                "time_ratio": case["time_ratio"],
            }
            for case in evaluation.get("cases") or []
        ],
    }


def write_ledger(path: Path, records: list[dict], factors: dict, status: str) -> dict:
    usage = empty_usage()
    unmetered = 0
    models: dict[str, int] = {}
    for record in records:
        models[record.get("model") or "unknown"] = models.get(record.get("model") or "unknown", 0) + 1
        if record.get("usage"):
            _add_usage(usage, record["usage"])
        else:
            unmetered += 1
    ledger = {
        "run_id": records[0]["run_id"] if records else None,
        "status": status,
        "factors": factors,
        "models": models,
        "calls": len(records),
        "metered_calls": len(records) - unmetered,
        "unmetered_calls": unmetered,
        "usage": usage,
        "generations": [
            {
                "generation": generation,
                "calls": sum(1 for record in records if record["generation"] == generation),
                "best": best_summary(
                    best_record([record for record in records if record["generation"] == generation])
                ),
            }
            for generation in sorted({record["generation"] for record in records})
        ],
        "best": best_summary(best_record(records)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger


def write_ablation_ledger(path: Path, ledgers: list[dict]) -> dict:
    usage = empty_usage()
    groups = []
    for ledger in ledgers:
        _add_usage(usage, ledger["usage"])
        groups.append(
            {
                "run_id": ledger["run_id"],
                "status": ledger["status"],
                "factors": ledger["factors"],
                "calls": ledger["calls"],
                "unmetered_calls": ledger["unmetered_calls"],
                "usage": ledger["usage"],
                "best": ledger["best"],
            }
        )
    payload = {
        "cells": len(groups),
        "calls": sum(item["calls"] for item in groups),
        "unmetered_calls": sum(item["unmetered_calls"] for item in groups),
        "usage": usage,
        "by_cell": groups,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
