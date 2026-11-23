"""Command line entry point: ``python3 -m harness_engineering.cli run ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .archive import archive_acceptance
from .models import MethodSpec
from .runner import evaluate_candidate, run_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one harness task")
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run", help="execute and evaluate a task JSON")
    run_parser.add_argument("task", type=Path)
    run_parser.add_argument("--workdir", type=Path, default=Path("."))
    run_parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.action == "run":
        spec = MethodSpec.from_dict(json.loads(args.task.read_text(encoding="utf-8")))
        run = run_candidate(spec, args.workdir, report_dir=args.report_dir)
        decision = evaluate_candidate(spec, run)
        result = {
            "task": spec.to_dict(),
            "run": run.to_dict(),
            "decision": decision.to_dict(),
        }
        output = json.dumps(result, indent=2, sort_keys=True)
        if args.report_dir is not None:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            acceptance_path = args.report_dir / f"{spec.name}.acceptance.json"
            acceptance_text = output + "\n"
            acceptance_path.write_text(acceptance_text, encoding="utf-8")
            archive_acceptance(
                args.report_dir,
                acceptance_text,
                task_name=spec.name,
                command=run.command,
                status=run.status,
                failure_kind=run.failure_kind,
                duration_s=run.duration_s,
                passed=decision.passed,
                reasons=decision.reasons,
                metrics=decision.metrics,
            )
        print(output)
        return 0 if decision.passed else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
