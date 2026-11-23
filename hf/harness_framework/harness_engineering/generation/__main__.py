"""Public run/resume/evaluate entry points for method-generation campaigns."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .campaign import Campaign
from .storage import load_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["single"]:
        # One generated program, small-molecule CPU accuracy gate only.
        from .single import main as single_main
        return single_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Generate RHF from an empty interface and independently verify it",
        epilog="single: generate ONE program and accept it on small-molecule PySCF accuracy only "
               "(see 'single --help')")
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("run", help="Start a new, frozen-spec campaign")
    defaults = Path(__file__).resolve().parents[2] / "harness/generation"
    run.add_argument("--task", type=Path, default=defaults / "rhf.json")
    run.add_argument("--holdout", type=Path, default=defaults / "holdout_rhf_v1.json")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    run.add_argument("--reasoning-effort", default=None)
    run.add_argument("--executor-config", type=Path, required=True)
    run.add_argument("--api-base-url", default="https://api.openai.com/v1")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--api-timeout", type=float, default=600,
                     help="Absolute API timeout; standard <=600, explicitly extended <=1200 seconds")
    run.add_argument("--budget-profile", choices=("standard", "extended"), default="standard",
                     help="Freeze an explicitly authorized budget profile; defaults remain standard")
    run.add_argument("--max-output-tokens", type=int, default=16384,
                     help="Total output including reasoning; standard <=16384, extended <=32768")
    run.add_argument("--stream", action="store_true",
                     help="Receive bounded Responses SSE; accept only a complete response.completed event")
    for operation in ("resume", "evaluate"):
        command = sub.add_parser(operation)
        command.add_argument("--campaign", type=Path, required=True)
        if operation == "resume":
            command.add_argument("--enable-gpu", action="store_true",
                                 help="Explicitly authorize GPU work previously stopped by configuration")
    args = parser.parse_args(argv)
    try:
        if args.operation == "run":
            state = Campaign(args.output).run(
                args.task, args.holdout, model=args.model, reasoning_effort=args.reasoning_effort,
                execution_config=load_json(args.executor_config),
                generation_config={"base_url": args.api_base_url, "api_key_env": args.api_key_env,
                                   "budget_profile": args.budget_profile,
                                   "max_output_tokens": args.max_output_tokens, "max_retries": 2, "timeout_s": args.api_timeout,
                                   "streaming": args.stream},
            )
            print(json.dumps({key: state.get(key) for key in (
                "status", "phase", "round", "max_rounds", "blocked_reason", "accepted_candidate", "api_usage")}, indent=2))
            return 0 if state["status"] == "accepted" else 2 if state["status"].startswith("waiting_for_") else 1
        campaign = Campaign(args.campaign)
        if args.operation == "resume":
            state = campaign.resume(enable_gpu=args.enable_gpu)
            print(json.dumps({"status": state["status"], "phase": state["phase"], "round": state["round"],
                              "blocked_reason": state.get("blocked_reason")}, indent=2))
            return 0 if state["status"] == "accepted" else 2 if state["status"].startswith("waiting_for_") else 1
        result = campaign.evaluate()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["accepted"] else 1
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
