"""Run the oracle task before the budget task and apply the stop rules.

The loop does not call a model and does not edit ``hf.py``. A budget rejection
whose reasons are only diagonalization-projection overruns keeps the current
generated file as the last correct candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECTION_PREFIX = "projected_eigh_hours_"


def is_projection_reason(reason: str) -> bool:
    """True when a gate reason is only a projected-hours overrun."""

    key, separator, _rest = reason.partition("=")
    return bool(separator) and key.startswith(PROJECTION_PREFIX) and "exceeds tolerance" in reason


def correct_despite_projection(reasons: Sequence[str]) -> bool:
    """A failed decision is still a correct candidate when every reason is a projection."""

    return bool(reasons) and all(is_projection_reason(reason) for reason in reasons)


def syntax_or_interface(failure_kind: str | None, reasons: Sequence[str]) -> bool:
    if failure_kind == "syntax_error":
        return True
    text = " ".join(reasons).lower()
    return any(token in text for token in ("syntaxerror", "typeerror", "attributeerror"))


def numerical_error(failure_kind: str | None, reasons: Sequence[str]) -> bool:
    if failure_kind == "numerical_failure":
        return True
    return any(
        "exceeds tolerance" in reason and not is_projection_reason(reason)
        for reason in reasons
    )


def worst_projection(metrics: Mapping[str, Any]) -> float | None:
    values = [
        float(value)
        for key, value in metrics.items()
        if key.startswith(PROJECTION_PREFIX) and isinstance(value, (int, float))
    ]
    if not values:
        return None
    return max(values)


def projections_stalled(history: Sequence[float], rounds: int = 2) -> bool:
    """True after ``rounds`` consecutive projections that did not decrease."""

    if rounds < 1 or len(history) < rounds + 1:
        return False
    window = list(history[-(rounds + 1):])
    return all(window[index] >= window[index - 1] for index in range(1, len(window)))


def decide_round(
    *,
    oracle_passed: bool,
    oracle_failure_kind: str | None = None,
    oracle_reasons: Sequence[str] = (),
    budget_passed: bool | None = None,
    budget_failure_kind: str | None = None,
    budget_reasons: Sequence[str] = (),
    syntax_fixes: int = 0,
    projection_history: Sequence[float] = (),
    projection_value: float | None = None,
    stall_rounds: int = 2,
    forbidden_changed: Sequence[str] = (),
) -> dict[str, Any]:
    """Choose the next generation action without launching a candidate."""

    if forbidden_changed:
        return {
            "action": "stop",
            "run_budget": False,
            "correct_candidate": False,
            "reasons": [f"forbidden file changed: {path}" for path in forbidden_changed],
        }
    if not oracle_passed:
        if syntax_or_interface(oracle_failure_kind, oracle_reasons) and syntax_fixes < 1:
            action = "fix_once"
        elif numerical_error(oracle_failure_kind, oracle_reasons):
            action = "revert"
        else:
            action = "stop_before_budget"
        return {
            "action": action,
            "run_budget": False,
            "correct_candidate": False,
            "reasons": list(oracle_reasons),
        }
    if budget_passed is None:
        return {
            "action": "run_budget",
            "run_budget": True,
            "correct_candidate": False,
            "reasons": [],
        }
    if budget_passed:
        return {
            "action": "accept",
            "run_budget": True,
            "correct_candidate": True,
            "reasons": [],
        }
    reasons = list(budget_reasons)
    history = list(projection_history)
    if projection_value is not None:
        history.append(float(projection_value))
    if correct_despite_projection(reasons):
        action = "stop" if projections_stalled(history, stall_rounds) else "keep_correct_candidate"
        return {
            "action": action,
            "run_budget": True,
            "correct_candidate": True,
            "reasons": reasons,
        }
    if syntax_or_interface(budget_failure_kind, reasons) and syntax_fixes < 1:
        action = "fix_once"
    elif syntax_or_interface(budget_failure_kind, reasons):
        action = "stop"
    elif numerical_error(budget_failure_kind, reasons):
        action = "revert"
    else:
        action = "stop"
    return {
        "action": action,
        "run_budget": True,
        "correct_candidate": False,
        "reasons": reasons,
    }


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_files(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    names = sorted(set(before) | set(after))
    return [name for name in names if before.get(name) != after.get(name)]


def snapshot_files(root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    digests = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            digests[relative] = file_digest(path)
    return digests


def _load_task(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _acceptance_path(report_dir: Path, task_name: str) -> Path:
    return report_dir / f"{task_name}.acceptance.json"


def run_loop(
    oracle_task: Path,
    budget_task: Path,
    workdir: Path,
    report_dir: Path,
    *,
    syntax_fixes: int = 0,
    projection_history: Sequence[float] = (),
) -> dict[str, Any]:
    """Run the oracle, then the budget only if the oracle passes."""

    from .cli import main as cli_main

    workdir = Path(workdir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    oracle_raw = _load_task(oracle_task)
    budget_raw = _load_task(budget_task)
    generation = dict(oracle_raw.get("generation", {}))
    forbidden = list(generation.get("forbidden", ()))
    stall_rounds = int(generation.get("stop", {}).get("stall_rounds", 2))
    before = snapshot_files(workdir, forbidden)

    oracle_code = cli_main([
        "run", str(oracle_task),
        "--workdir", str(workdir),
        "--report-dir", str(report_dir),
    ])
    oracle_acceptance = json.loads(
        _acceptance_path(report_dir, oracle_raw["name"]).read_text(encoding="utf-8"))
    oracle_decision = oracle_acceptance["decision"]
    oracle_run = oracle_acceptance["run"]
    forbidden_changed = changed_files(before, snapshot_files(workdir, forbidden))
    oracle_passed = bool(oracle_decision["passed"]) and oracle_code == 0 and not forbidden_changed
    decision = decide_round(
        oracle_passed=oracle_passed,
        oracle_failure_kind=oracle_run.get("failure_kind"),
        oracle_reasons=oracle_decision.get("reasons", ()),
        syntax_fixes=syntax_fixes,
        stall_rounds=stall_rounds,
        forbidden_changed=forbidden_changed,
    )
    budget_acceptance = None
    if decision["run_budget"]:
        before_budget = snapshot_files(workdir, forbidden)
        budget_code = cli_main([
            "run", str(budget_task),
            "--workdir", str(workdir),
            "--report-dir", str(report_dir),
        ])
        budget_acceptance = json.loads(
            _acceptance_path(report_dir, budget_raw["name"]).read_text(encoding="utf-8"))
        forbidden_changed = changed_files(
            before_budget, snapshot_files(workdir, forbidden))
        metrics = budget_acceptance["decision"].get("metrics", {})
        decision = decide_round(
            oracle_passed=True,
            budget_passed=bool(budget_acceptance["decision"]["passed"]) and budget_code == 0,
            budget_failure_kind=budget_acceptance["run"].get("failure_kind"),
            budget_reasons=budget_acceptance["decision"].get("reasons", ()),
            syntax_fixes=syntax_fixes,
            projection_history=projection_history,
            projection_value=worst_projection(metrics),
            stall_rounds=stall_rounds,
            forbidden_changed=forbidden_changed,
        )
    summary = {
        "action": decision["action"],
        "correct_candidate": decision["correct_candidate"],
        "run_budget": decision["run_budget"],
        "reasons": decision["reasons"],
        "oracle_passed": oracle_passed,
        "oracle_task": oracle_raw["name"],
        "budget_task": budget_raw["name"] if budget_acceptance is not None else None,
        "budget_passed": (
            None if budget_acceptance is None else bool(budget_acceptance["decision"]["passed"])
        ),
        "editable": list(generation.get("editable", ())),
        "forbidden": forbidden,
    }
    summary_path = report_dir / "occupy-density-loop.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the occupy-density generation loop once")
    parser.add_argument("--oracle", type=Path, default=Path("harness/tasks/occupy_density_oracle.json"))
    parser.add_argument("--budget", type=Path, default=Path("harness/tasks/occupy_density_budget.json"))
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args(argv)
    summary = run_loop(args.oracle, args.budget, args.workdir, args.report_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["action"] in {"accept", "keep_correct_candidate"}:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
