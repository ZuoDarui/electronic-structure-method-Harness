"""Append-only archive of harness acceptance decisions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def archive_acceptance(
    report_dir: str | Path,
    acceptance_text: str,
    *,
    task_name: str,
    command: Sequence[str],
    status: str,
    failure_kind: str | None,
    duration_s: float,
    passed: bool,
    reasons: Sequence[str],
    metrics: Mapping[str, Any],
    when: datetime | None = None,
) -> dict[str, Any]:
    """Save one immutable acceptance copy and append a one-line record.

    The latest ``<task>.acceptance.json`` written by the CLI can be replaced.
    The copy under ``runs/`` and the ``experiments.jsonl`` line are not.
    """

    root = Path(report_dir)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    candidate_id = _unique_candidate_id(runs, task_name, when or _utcnow())
    copy_path = runs / f"{candidate_id}.acceptance.json"
    text = acceptance_text if acceptance_text.endswith("\n") else acceptance_text + "\n"
    copy_path.write_text(text, encoding="utf-8")
    record = {
        "acceptance_path": str(copy_path.resolve()),
        "candidate_id": candidate_id,
        "command": list(command),
        "duration_s": duration_s,
        "failure_kind": failure_kind,
        "metrics": scalar_metrics(metrics),
        "passed": bool(passed),
        "reasons": list(reasons),
        "status": status,
        "task": task_name,
    }
    jsonl_path = root / "experiments.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def backfill_acceptance(report_dir: str | Path, acceptance_path: str | Path) -> dict[str, Any] | None:
    """Archive an existing acceptance file once.

    A second call with the same task and duration does not append another line.
    """

    path = Path(acceptance_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["task"]
    run = payload["run"]
    decision = payload["decision"]
    task_name = str(task["name"])
    duration_s = float(run["duration_s"])
    root = Path(report_dir)
    if _already_recorded(root / "experiments.jsonl", task_name, duration_s):
        return None
    when = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return archive_acceptance(
        root,
        path.read_text(encoding="utf-8"),
        task_name=task_name,
        command=run.get("command", ()),
        status=str(run.get("status", "")),
        failure_kind=run.get("failure_kind"),
        duration_s=duration_s,
        passed=bool(decision.get("passed")),
        reasons=decision.get("reasons", ()),
        metrics=decision.get("metrics", {}),
        when=when,
    )


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only JSON scalars used by the acceptance gate."""

    kept: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            kept[str(key)] = value
    return kept


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _unique_candidate_id(runs: Path, task_name: str, when: datetime) -> str:
    stamp = when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{task_name}-{stamp}"
    candidate_id = base
    suffix = 2
    while (runs / f"{candidate_id}.acceptance.json").exists():
        candidate_id = f"{base}-{suffix}"
        suffix += 1
    return candidate_id


def _already_recorded(jsonl_path: Path, task_name: str, duration_s: float) -> bool:
    if not jsonl_path.exists():
        return False
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task") == task_name and record.get("duration_s") == duration_s:
            return True
    return False
