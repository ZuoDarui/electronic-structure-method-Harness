"""Command line for search sessions, ablation grids, and the baseline gate.

A search is driven one generation at a time::

    esm-harness init --config configs/hf_rhf.yaml
    esm-harness step --session runs/hf-opus55     # lists prompts awaiting a model
    # the driver answers each prompt with Claude Opus 5.5, writes the response,
    # then records who answered:
    esm-harness meta --session runs/hf-opus55 --cid <cid> --model claude-opus-5-5-high --agent-id <id>
    esm-harness step --session runs/hf-opus55     # scores and writes the next prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esm_harness.config import REPO_ROOT, cell_id, expand_grid, load_yaml, resolve
from esm_harness.eval.cases import load_cases
from esm_harness.eval.reference import baseline_errors, pyscf_energies, read_references, write_references
from esm_harness.eval.submit import make_submitter
from esm_harness.search.session import Session, _now
from esm_harness.trace import append_record, write_ablation_ledger


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _session_dir(config: dict, out: str | None) -> Path:
    if out:
        return resolve(out)
    root = resolve(config.get("out_dir") or "runs")
    return root / str(config["run_id"]) if config.get("run_id") else root


def _cmd_init(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    session = Session(_session_dir(config, args.out))
    session.init(config, make_submitter(config, REPO_ROOT))
    _print(session.status())
    return 0


def _cmd_step(args: argparse.Namespace) -> int:
    session = Session(resolve(args.session))
    config = session.load()["config"]
    _print(session.advance(make_submitter(config, REPO_ROOT)))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _print(Session(resolve(args.session)).status())
    return 0


def _session_for(cid: str, session: str | None) -> Session:
    if session:
        return Session(resolve(session))
    for state in resolve("trajectories").glob("**/session.json"):
        if cid.startswith(json.loads(state.read_text(encoding="utf-8"))["run_id"] + "-g"):
            return Session(state.parent)
    raise SystemExit(f"no session under trajectories/ owns {cid}; pass --session")


def _cmd_meta(args: argparse.Namespace) -> int:
    session = _session_for(args.cid, args.session)
    usage = json.loads(args.usage_json) if args.usage_json else None
    meta = {
        "provider": "cursor-subagent",
        "model": args.model,
        "agent_id": args.agent_id,
        "usage": usage,
        "usage_note": None if usage else "the subagent call did not report token usage",
        "latency_s": args.latency_s,
    }
    append_record(
        session.root / "calls.jsonl",
        {"candidate_id": args.cid, "at": _now(), "error": args.error, **meta},
    )
    if not args.error:
        session.path("meta", args.cid).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _print({"candidate_id": args.cid, "error": args.error, **meta})
    return 0


def _cells(grid_path: str) -> tuple[dict, list[tuple[str, dict]]]:
    document = load_yaml(grid_path)
    cells = expand_grid(document)
    return document, [(cell_id(index, cell["factors"]), cell) for index, cell in enumerate(cells)]


def _cmd_ablate_init(args: argparse.Namespace) -> int:
    document, cells = _cells(args.grid)
    root = resolve(document.get("out_dir") or "runs/ablation")
    out = []
    for name, cell in cells:
        session = Session(root / name)
        if not session.exists():
            cell = dict(cell)
            cell["run_id"] = name
            session.init(cell, make_submitter(cell, REPO_ROOT))
        out.append(session.status())
    _print(out)
    return 0


def _cmd_ablate_step(args: argparse.Namespace) -> int:
    document, cells = _cells(args.grid)
    root = resolve(document.get("out_dir") or "runs/ablation")
    statuses = []
    for name, _ in cells:
        session = Session(root / name)
        config = session.load()["config"]
        statuses.append(session.advance(make_submitter(config, REPO_ROOT)))
    if all(item["status"] == "done" for item in statuses):
        ledgers = [json.loads((root / name / "ledger.json").read_text(encoding="utf-8")) for name, _ in cells]
        write_ablation_ledger(root / "ablation-ledger.json", ledgers)
    _print(statuses)
    return 0


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _cmd_report(args: argparse.Namespace) -> int:
    root = resolve(args.root)
    lines = [
        "| run | prompt | search | speed_weight | effort | calls | failed calls | best gen | "
        "best fitness | max err vs PySCF | max err vs baseline | geomean time ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for state_path in sorted(root.glob("**/session.json")):
        session = Session(state_path.parent)
        state = session.load()
        factors = state["factors"]
        ledger = json.loads(session.ledger_path.read_text(encoding="utf-8")) if session.ledger_path.is_file() else {}
        calls_log = session.root / "calls.jsonl"
        failed = 0
        if calls_log.is_file():
            failed = sum(1 for line in calls_log.read_text(encoding="utf-8").splitlines() if json.loads(line).get("error"))
        best = ledger.get("best") or {}
        rows.append({"run_id": state["run_id"], "status": state["status"], "factors": factors, "best": best})
        lines.append(
            f"| {state['run_id']} ({state['status']}) | {factors['prompt_id']} | {factors['search']} | "
            f"{factors['speed_weight']} | {factors['reasoning_effort']} | {ledger.get('calls', 0)} | {failed} | "
            f"{best.get('generation', '-')} | {_fmt(best.get('fitness'), '.3f')} | "
            f"{_fmt(best.get('max_abs_err'), '.1e')} | {_fmt(best.get('max_abs_err_vs_baseline'), '.1e')} | "
            f"{_fmt(best.get('geomean_time_ratio'), '.4f')} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    from esm_harness.rescore import rescore_session, write_rescore_report

    root = resolve(args.root)
    baseline = resolve(args.baseline)
    results = []
    for state_path in sorted(root.glob("**/session.json")):
        session = Session(state_path.parent)
        config = session.load()["config"]
        results.append(rescore_session(session, baseline, make_submitter(config, REPO_ROOT)))
    print(write_rescore_report(root, baseline, results).read_text(encoding="utf-8"))
    return 0


def _audit_inputs(args: argparse.Namespace, sessions: list[Session]):
    from esm_harness.audit import load_agent_transcripts, load_launches

    if args.driver_transcript:
        launch_files = [resolve(path) for path in args.driver_transcript]
    else:
        launch_files = [s.root / "launches.jsonl" for s in sessions if (s.root / "launches.jsonl").is_file()]
    transcript_dirs = [resolve(path) for path in args.transcripts] or [s.root for s in sessions]
    return load_launches(launch_files), load_agent_transcripts(transcript_dirs)


def _sessions_under(root: Path) -> list[Session]:
    return [Session(path.parent) for path in sorted(root.glob("**/session.json"))]


def _cmd_audit(args: argparse.Namespace) -> int:
    from esm_harness.audit import archive, audit_session, reused_agents, write_audit, write_audit_report

    root = resolve(args.root)
    sessions = _sessions_under(root)
    launches, transcripts = _audit_inputs(args, sessions)
    if not launches:
        raise SystemExit("no subagent launches found; pass --driver-transcript or audit an archived session")
    reused = reused_agents(launches)
    results = []
    for session in sessions:
        result = audit_session(session, launches, transcripts, reused)
        if args.archive:
            archive(session, result, transcripts)
        write_audit(session, result)
        results.append(result)
    print(write_audit_report(root, results).read_text(encoding="utf-8"))
    violations = sum(result["summary"]["violations"] for result in results)
    return 1 if args.strict and violations else 0


def _cmd_audit_ids(args: argparse.Namespace) -> int:
    from esm_harness.audit import load_launches

    run_ids = {s.load()["run_id"] for s in _sessions_under(resolve(args.root))}
    ids = [
        launch.agent_id
        for launch in load_launches([resolve(path) for path in args.driver_transcript])
        if launch.cid and any(launch.cid.startswith(run_id + "-g") for run_id in run_ids)
    ]
    for start in range(0, len(ids), 50):
        print(json.dumps(ids[start : start + 50]))
    return 0


def _cmd_reference(args: argparse.Namespace) -> int:
    cases = load_cases(resolve(args.suite))
    _print(write_references(cases, resolve(args.out)))
    return 0


def _cmd_check_baseline(args: argparse.Namespace) -> int:
    cases = load_cases(resolve(args.suite))
    fresh = pyscf_energies(cases)
    reference_path = resolve(args.references)
    if reference_path.is_file():
        cached = read_references(reference_path, cases)
        for case_id, energy in fresh.items():
            if abs(energy - cached[case_id]) > 1e-10:
                print(f"{case_id}: cached reference {cached[case_id]} disagrees with PySCF {energy}", file=sys.stderr)
                return 1
        compared = cached
    else:
        compared = fresh
    failed = False
    for row in baseline_errors(resolve(args.baseline), cases, compared):
        print(f"{row['id']}: energy={row['energy']:.12f} abs_err={row['abs_err']:.3e} converged={row['converged']}")
        if not row["converged"] or row["abs_err"] > args.tol:
            failed = True
    if failed:
        print(f"baseline is outside {args.tol}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esm-harness")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="measure the baseline and write the generation-0 prompts")
    init.add_argument("--config", required=True)
    init.add_argument("--out", help="session directory (default: out_dir/run_id from the config)")
    init.set_defaults(func=_cmd_init)

    step = commands.add_parser("step", help="score answered prompts and write the next generation")
    step.add_argument("--session", required=True)
    step.set_defaults(func=_cmd_step)

    status = commands.add_parser("status", help="show a session's progress and pending prompts")
    status.add_argument("--session", required=True)
    status.set_defaults(func=_cmd_status)

    meta = commands.add_parser("meta", help="record which model answered a prompt and its token usage")
    meta.add_argument("--session", help="session directory (default: found from the candidate id)")
    meta.add_argument("--cid", required=True)
    meta.add_argument("--error", help="the call failed with this message; it is logged and retried")
    meta.add_argument("--model", required=True)
    meta.add_argument("--agent-id")
    meta.add_argument("--usage-json", help="token usage reported by the call; omit when none was reported")
    meta.add_argument("--latency-s", type=float)
    meta.set_defaults(func=_cmd_meta)

    ablate_init = commands.add_parser("ablate-init", help="create one session per ablation cell")
    ablate_init.add_argument("--grid", required=True)
    ablate_init.set_defaults(func=_cmd_ablate_init)

    ablate_step = commands.add_parser("ablate-step", help="advance every ablation cell")
    ablate_step.add_argument("--grid", required=True)
    ablate_step.set_defaults(func=_cmd_ablate_step)

    report = commands.add_parser("report", help="tabulate the best result of every session under a directory")
    report.add_argument("--root", default="trajectories")
    report.set_defaults(func=_cmd_report)

    rescore = commands.add_parser("rescore", help="re-time every recorded program against another baseline")
    rescore.add_argument("--root", default="trajectories")
    rescore.add_argument("--baseline", default="baselines/pyscf_rhf.py")
    rescore.set_defaults(func=_cmd_rescore)

    audit = commands.add_parser("audit", help="check sessions against the driver's and subagents' transcripts")
    audit.add_argument("--root", default="trajectories")
    audit.add_argument(
        "--driver-transcript",
        action="append",
        default=[],
        help="transcript JSON of the agent that launched the subagents (default: archived launches.jsonl)",
    )
    audit.add_argument(
        "--transcripts",
        action="append",
        default=[],
        help="directory holding <agent-id>/transcript.json files (default: archived agent_transcripts/)",
    )
    audit.add_argument("--archive", action="store_true", help="copy the launches and transcripts used into each session")
    audit.add_argument("--strict", action="store_true", help="exit 1 when any violation is found")
    audit.set_defaults(func=_cmd_audit)

    audit_ids = commands.add_parser("audit-ids", help="list subagent ids to fetch transcripts for, 50 per line")
    audit_ids.add_argument("--root", default="trajectories")
    audit_ids.add_argument("--driver-transcript", action="append", required=True)
    audit_ids.set_defaults(func=_cmd_audit_ids)

    reference = commands.add_parser("reference", help="cache PySCF RHF energies for the suite")
    reference.add_argument("--suite", default="benchmarks/hf_rhf.yaml")
    reference.add_argument("--out", default="benchmarks/reference_energies.json")
    reference.set_defaults(func=_cmd_reference)

    check = commands.add_parser("check-baseline", help="require the baseline to match PySCF")
    check.add_argument("--suite", default="benchmarks/hf_rhf.yaml")
    check.add_argument("--baseline", default="baselines/pyscf_rhf.py")
    check.add_argument("--references", default="benchmarks/reference_energies.json")
    check.add_argument("--tol", type=float, default=1e-8)
    check.set_defaults(func=_cmd_check_baseline)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except BrokenPipeError:
        code = 0
    raise SystemExit(code)
