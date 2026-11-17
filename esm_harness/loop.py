"""One-round lifecycle for the driving agent, and a read-only view for a monitor.

Each round is driven by a fresh top-level agent that lives for that round only
(see ``LOOP.md``):

    round begin --agent-id <id>   lock the harness, open every planned session
    (one fresh subagent per open request, ``meta`` after each)
    round end --agent-id <id>     score, write the next prompts, write the
                                  round report, release the lock

While ``LOOP_LOCK.json`` exists the harness is frozen: the lock records a
fingerprint of every harness file, and ``round end`` and ``step`` refuse to
score if it changed. Harness edits belong in the time between rounds.

``snapshot`` only reads, so a monitoring agent can watch a round in progress
without touching it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from esm_harness.config import REPO_ROOT, cell_id, expand_grid, load_yaml, resolve
from esm_harness.eval.submit import make_submitter
from esm_harness.search.session import Session, _now
from esm_harness.trace import read_records

LOCK_NAME = "LOOP_LOCK.json"
HARNESS_PATHS = (
    "esm_harness",
    "configs",
    "ablations",
    "baselines",
    "benchmarks",
    "tests",
    "pyproject.toml",
    "loop.yaml",
    "LOOP.md",
)


class LoopError(RuntimeError):
    pass


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in HARNESS_PATHS:
        path = root / rel
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
            )
        else:
            files = []
        for item in files:
            digest.update(item.relative_to(root).as_posix().encode() + b"\0" + item.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def read_lock(root: Path) -> dict | None:
    path = root / LOCK_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def planned_sessions(plan: dict) -> list[tuple[Session, dict]]:
    out = []
    for entry in plan.get("sessions") or []:
        if "config" in entry:
            config = load_yaml(resolve(entry["config"]))
            if entry.get("run_id"):
                config["run_id"] = entry["run_id"]
            out.append((Session(resolve(config.get("out_dir") or "runs") / str(config["run_id"])), config))
        elif "grid" in entry:
            document = load_yaml(resolve(entry["grid"]))
            root = resolve(document.get("out_dir") or "runs/ablation")
            for index, cell in enumerate(expand_grid(document)):
                name = cell_id(index, cell["factors"])
                out.append((Session(root / name), {**cell, "run_id": name}))
        else:
            raise LoopError(f"loop plan entry needs 'config' or 'grid': {entry}")
    return out


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _rounds_dir(root: Path, plan: dict) -> Path:
    return root / str(plan.get("rounds_dir") or "trajectories/rounds")


def _locked_sessions(root: Path, lock: dict) -> list[Session]:
    return [Session(root / path) for path in lock["sessions"]]


def open_requests(root: Path) -> list[dict]:
    lock = read_lock(root)
    if lock is None:
        return []
    out = []
    for session in _locked_sessions(root, lock):
        for request in session.requests():
            out.append({"session": _rel(root, session.root), **request})
    return out


def begin(root: Path, plan: dict, agent_id: str) -> dict:
    if plan.get("paused"):
        raise LoopError("the loop is paused (loop.yaml: paused: true); no round was started")
    lock = read_lock(root)
    if lock is not None:
        if lock["driver_agent_id"] == agent_id:
            return {**lock, "requests": open_requests(root), "resumed": True}
        raise LoopError(
            f"round {lock['round']} is held by {lock['driver_agent_id']} since {lock['started']}; "
            "only that agent may continue it"
        )
    started = _now()
    active, fresh = [], []
    for session, config in planned_sessions(plan):
        submitter = make_submitter(config, REPO_ROOT)
        if getattr(submitter, "asynchronous", False):
            try:
                submitter.preflight()
            except Exception as exc:
                raise LoopError(f"{config['run_id']}: the cluster cannot take jobs: {exc}") from exc
        if not session.exists():
            session.init(config, submitter, run_id=config["run_id"])
            fresh.append(_rel(root, session.root))
        if session.load()["status"] == "evaluating":
            session.advance(submitter)
        if session.load()["status"] != "done":
            active.append(_rel(root, session.root))
    if not active:
        raise LoopError("every planned session is done; nothing to run")
    if not any(Session(root / path).requests() for path in active):
        waiting = [path for path in active if Session(root / path).load()["status"] == "evaluating"]
        if waiting:
            raise LoopError(
                f"waiting for cluster evaluations in {', '.join(waiting)}; no round was started"
            )
    number = len(list(_rounds_dir(root, plan).glob("round-*.json"))) + 1
    lock = {
        "round": number,
        "driver_agent_id": agent_id,
        "started": started,
        "fingerprint": fingerprint(root),
        "sessions": active,
        "baseline_measured_at_begin": fresh,
    }
    (root / LOCK_NAME).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {**lock, "requests": open_requests(root), "resumed": False}


def check_frozen(root: Path) -> None:
    lock = read_lock(root)
    if lock is not None and fingerprint(root) != lock["fingerprint"]:
        raise LoopError(
            f"harness files changed during round {lock['round']} (fingerprint {fingerprint(root)}, "
            f"locked at {lock['fingerprint']}); restore them before scoring"
        )


def end(root: Path, plan: dict, agent_id: str) -> dict:
    lock = read_lock(root)
    if lock is None:
        raise LoopError("no round is open")
    if lock["driver_agent_id"] != agent_id:
        raise LoopError(f"round {lock['round']} belongs to {lock['driver_agent_id']}, not {agent_id}")
    check_frozen(root)
    waiting = open_requests(root)
    if waiting:
        raise LoopError(f"{len(waiting)} requests are still open: {', '.join(r['candidate_id'] for r in waiting)}")
    sessions = []
    for session in _locked_sessions(root, lock):
        before = session.load()
        submitter = make_submitter(before["config"], REPO_ROOT)
        if _rel(root, session.root) not in lock.get("baseline_measured_at_begin", []):
            session.remeasure_baseline(submitter)
        baseline = session.baseline()
        scored_before = len(read_records(session.trajectory))
        status = session.advance(submitter)
        scored = read_records(session.trajectory)[scored_before:]
        sessions.append(
            {
                "session": _rel(root, session.root),
                "run_id": status["run_id"],
                "generation_scored": before["generation"],
                "baseline": {"measured": baseline["measured"], "host": baseline.get("host")},
                "status": status["status"],
                "next_generation": status["generation"]
                if status["status"] not in ("done", "evaluating")
                else None,
                "evaluating": [
                    {
                        "candidate_id": cid,
                        "tasks": [
                            {"case_id": job["case_id"], "task_id": job["handle"].get("task_id"),
                             "submit_error": job["handle"].get("submit_error")}
                            for job in started.get("jobs", [])
                        ],
                    }
                    for cid, started in sorted((session.load().get("evaluating") or {}).items())
                ],
                "candidates": [
                    {
                        "candidate_id": record["candidate_id"],
                        "agent_id": record.get("agent_id"),
                        "status": record["eval"].get("status"),
                        "fitness": record["eval"].get("fitness"),
                        "max_abs_err": record["eval"].get("max_abs_err"),
                        "geomean_time_ratio": record["eval"].get("geomean_time_ratio"),
                    }
                    for record in scored
                ],
                "best": status["best"],
            }
        )
    report = {**lock, "ended": _now(), "sessions": sessions}
    out = _rounds_dir(root, plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"round-{lock['round']:03d}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / LOCK_NAME).unlink()
    return report


def _age(stamp: str) -> str:
    seconds = int((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stamp)).total_seconds())
    return f"{seconds // 60}m{seconds % 60:02d}s"


def snapshot(root: Path, plan: dict, launches: list | None = None, transcripts: dict | None = None) -> dict:
    """Everything a monitor needs about the loop, without writing anything."""
    from esm_harness.audit import inspect_subagent

    lock = read_lock(root)
    current = fingerprint(root)
    if lock is not None:
        sessions = _locked_sessions(root, lock)
    else:
        sessions = [session for session, _ in planned_sessions(plan)]
    by_cid: dict[str, list] = {}
    for launch in launches or []:
        if launch.cid:
            by_cid.setdefault(launch.cid, []).append(launch)
    views = []
    for session in sessions:
        if not session.exists():
            views.append({"session": _rel(root, session.root), "status": "not started"})
            continue
        state = session.load()
        calls = read_records(session.root / "calls.jsonl")
        pending = []
        for cid in state["pending"]:
            cid_calls = [call for call in calls if call["candidate_id"] == cid]
            seen = []
            for launch in by_cid.get(cid, []):
                child = (transcripts or {}).get(launch.agent_id)
                wrote = None
                if child is not None:
                    wrote = inspect_subagent(
                        child["messages"],
                        f"{session.root.name}/prompts/{cid}.prompt.md",
                        f"{session.root.name}/responses/{cid}.response.md",
                    )["wrote_response"]
                seen.append(
                    {
                        "agent_id": launch.agent_id,
                        "model": launch.model,
                        "model_ok": launch.model == state["subagent_model"],
                        "wrote_response": wrote,
                    }
                )
            pending.append(
                {
                    "candidate_id": cid,
                    "response": session.path("responses", cid).is_file(),
                    "meta": session.path("meta", cid).is_file(),
                    "logged_calls": len(cid_calls),
                    "logged_failures": sum(1 for call in cid_calls if call.get("error")),
                    "launches": seen,
                }
            )
        best = (json.loads(session.ledger_path.read_text(encoding="utf-8")).get("best") or {}) if session.ledger_path.is_file() else {}
        views.append(
            {
                "session": _rel(root, session.root),
                "run_id": state["run_id"],
                "status": state["status"],
                "generation": state["generation"],
                "subagent_model": state["subagent_model"],
                "candidates_scored": len(read_records(session.trajectory)),
                "best": {key: best.get(key) for key in ("candidate_id", "fitness", "max_abs_err", "geomean_time_ratio")},
                "pending": pending,
                "evaluating": {
                    cid: {
                        "started_at": started.get("started_at"),
                        "tasks": [job["handle"].get("task_id") for job in started.get("jobs", [])],
                    }
                    for cid, started in (state.get("evaluating") or {}).items()
                },
            }
        )
    rounds = []
    for path in sorted(_rounds_dir(root, plan).glob("round-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rounds.append(
            {
                "round": report["round"],
                "driver_agent_id": report["driver_agent_id"],
                "started": report["started"],
                "ended": report["ended"],
                "scored": sum(len(item["candidates"]) for item in report["sessions"]),
            }
        )
    return {
        "state": "active" if lock else "idle",
        "paused": bool(plan.get("paused")),
        "lock": None
        if lock is None
        else {**lock, "elapsed": _age(lock["started"]), "harness_unchanged": current == lock["fingerprint"]},
        "fingerprint": current,
        "sessions": views,
        "rounds": rounds,
    }


def format_snapshot(view: dict) -> str:
    lines = []
    lock = view["lock"]
    if lock:
        frozen = "unchanged" if lock["harness_unchanged"] else "CHANGED SINCE THE ROUND BEGAN"
        lines.append(
            f"loop: ACTIVE, round {lock['round']}, driver {lock['driver_agent_id']}, "
            f"started {lock['started']} ({lock['elapsed']} ago); harness {frozen}"
        )
    else:
        lines.append(f"loop: idle (harness fingerprint {view['fingerprint']}); harness edits are allowed")
    if view.get("paused"):
        lines.append("loop.yaml: paused; new rounds will not start")
    for item in view["sessions"]:
        if item.get("status") == "not started":
            lines.append(f"- {item['session']}: not started")
            continue
        best = item["best"]
        best_text = (
            f"best {best['candidate_id']} fitness {best['fitness']:.3f} ratio {best['geomean_time_ratio']:.4f}"
            if best.get("fitness") is not None
            else "no scored candidate yet"
        )
        lines.append(
            f"- {item['run_id']}: {item['status']}, generation {item['generation']}, "
            f"{item['candidates_scored']} scored, {best_text}"
        )
        for cid, info in (item.get("evaluating") or {}).items():
            lines.append(
                f"    {cid}: evaluating on the cluster since {info['started_at']}, "
                f"{len(info['tasks'])} tasks ({', '.join(str(task) for task in info['tasks'])})"
            )
        for cid in item["pending"]:
            launched = ", ".join(
                f"{launch['agent_id']} {launch['model']}"
                + ("" if launch["model_ok"] else " WRONG MODEL")
                + {True: " wrote", False: " no write yet", None: ""}[launch["wrote_response"]]
                for launch in cid["launches"]
            )
            lines.append(
                f"    {cid['candidate_id']}: response {'yes' if cid['response'] else 'no'}, "
                f"meta {'yes' if cid['meta'] else 'no'}, calls {cid['logged_calls']} "
                f"({cid['logged_failures']} failed)" + (f"; launches: {launched}" if launched else "")
            )
    for item in view["rounds"][-5:]:
        lines.append(
            f"round {item['round']}: {item['driver_agent_id']}, {item['started']} -> {item['ended']}, "
            f"{item['scored']} scored"
        )
    return "\n".join(lines)
