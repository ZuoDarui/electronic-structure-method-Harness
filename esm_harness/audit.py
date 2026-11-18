"""Check recorded sessions against the platform's own agent transcripts.

The harness only sees files, and ``meta``/``calls.jsonl`` are written by the
driver. The transcripts are written by the agent platform: the driver's
transcript records every subagent launch with the model it requested, and each
subagent's transcript records its instructions, tool calls, and the file it
created. ``audit`` cross-checks the three:

* every scored candidate was answered by its own freshly launched subagent,
  launched with the session's bound model and the standard instructions;
* the subagent only read its prompt and only wrote its response, and the
  response that was scored is byte-for-byte what it wrote;
* every launch for a candidate appears in ``calls.jsonl``, and no failed launch
  had in fact produced a response.

Transcripts are fetched outside the harness (for Cursor cloud agents with the
``batch-fetch-details`` tool); ``audit-ids`` lists the ids to fetch. With
``--archive`` the launches and subagent transcripts used are copied into each
session so the audit can be rerun from the repository alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from esm_harness.llm.prompts import subagent_instructions
from esm_harness.search.session import Session, _now
from esm_harness.trace import read_records

LAUNCH_TOOLS = {"task", "task_v2"}
READ_TOOLS = {"read_file"}
WRITE_TOOLS = {"search_replace", "write", "edit_file", "write_file"}
_PROMPT_RE = re.compile(r"(\S+/prompts/([^/\s]+)\.prompt\.md)")
_RESPONSE_RE = re.compile(r"(\S+/responses/([^/\s]+)\.response\.md)")


@dataclass
class Launch:
    agent_id: str
    model: str | None
    instructions: str
    prompt_path: str | None
    response_path: str | None
    cid: str | None
    order: int
    driver: str | None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _launch_from_message(message: dict, order: int, driver: str | None) -> Launch | None:
    if message.get("role") != "tool" or message.get("tool_name") not in LAUNCH_TOOLS:
        return None
    args = message.get("tool_args") or {}
    value = (message.get("tool_result") or {}).get("value") or {}
    agent_id = value.get("cloudAgentBcId") or value.get("agentId")
    if not agent_id:
        return None
    text = str(args.get("prompt") or "")
    prompt = _PROMPT_RE.search(text)
    response = _RESPONSE_RE.search(text)
    return Launch(
        agent_id=agent_id,
        model=args.get("model"),
        instructions=text,
        prompt_path=prompt.group(1) if prompt else None,
        response_path=response.group(1) if response else None,
        cid=prompt.group(2) if prompt else None,
        order=order,
        driver=driver,
    )


def load_launches(paths: list[Path]) -> list[Launch]:
    """Launches from driver transcripts (``{"messages": [...]}``) or archived ``launches.jsonl``."""
    launches: list[Launch] = []
    for path in paths:
        if path.suffix == ".jsonl":
            launches.extend(Launch(**row) for row in read_records(path))
            continue
        driver = path.parent.name if path.parent.name.startswith("bc-") else None
        messages = json.loads(path.read_text(encoding="utf-8")).get("messages") or []
        for order, message in enumerate(messages):
            launch = _launch_from_message(message, order, driver)
            if launch:
                launches.append(launch)
    return launches


def load_agent_transcripts(dirs: list[Path]) -> dict[str, dict]:
    """``{agent_id: {"created_at_ms", "messages"}}`` from fetch directories or archived copies."""
    created: dict[str, int | None] = {}
    found: dict[str, dict] = {}
    for root in dirs:
        for index in root.rglob("index.json"):
            for agent in json.loads(index.read_text(encoding="utf-8")).get("agents") or []:
                created[agent["bcId"]] = agent.get("createdAtMs")
        for path in root.rglob("transcript.json"):
            if path.parent.name.startswith("bc-"):
                data = json.loads(path.read_text(encoding="utf-8"))
                found[path.parent.name] = {"created_at_ms": None, "messages": data.get("messages") or []}
        for path in root.rglob("agent_transcripts/bc-*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            found[path.stem] = {"created_at_ms": data.get("created_at_ms"), "messages": data.get("messages") or []}
    for agent_id, entry in found.items():
        if entry["created_at_ms"] is None:
            entry["created_at_ms"] = created.get(agent_id)
    return found


def _created_file(diff: str) -> tuple[str, str] | None:
    """(path, text) when an edit diff creates a whole file, else None."""
    lines = diff.split("\n")
    if len(lines) < 2 or lines[0] != "--- /dev/null" or not lines[1].startswith("+++ "):
        return None
    path = lines[1][4:]
    if path.startswith("b/"):
        path = path[2:]
    body: list[str] = []
    newline_at_end = True
    for line in lines[2:]:
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            body.append(line[1:])
        elif line.startswith("\\"):
            newline_at_end = False
        elif line:
            return None
    text = "\n".join(body)
    return path, text + "\n" if newline_at_end and body else text


def _edit_target(message: dict) -> tuple[str | None, str | None]:
    """(path, created text) for a write tool result."""
    args = message.get("tool_args") or {}
    path = args.get("target_file") or args.get("file_path") or args.get("path")
    value = (message.get("tool_result") or {}).get("value") or {}
    chunks = ((value.get("diff") or {}).get("chunks")) or []
    if chunks:
        created = _created_file("\n".join(chunk.get("diffString", "") for chunk in chunks))
        if created:
            return path or created[0], created[1]
        header = chunks[0].get("diffString", "").split("\n")
        if len(header) > 1 and header[1].startswith("+++ "):
            path = path or header[1][4:].removeprefix("b/")
    return path, None


def _ends_with(path: str | None, tail: str) -> bool:
    return bool(path) and Path(path).as_posix().endswith(tail)


def inspect_subagent(messages: list[dict], prompt_tail: str, response_tail: str) -> dict:
    """What a subagent did, judged against its one-read, one-write instructions."""
    violations: list[str] = []
    notes: list[str] = []
    written: str | None = None
    wrote_response = False
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool = message.get("tool_name")
        if tool in READ_TOOLS:
            target = (message.get("tool_args") or {}).get("target_file")
            if _ends_with(target, prompt_tail):
                continue
            if _ends_with(target, response_tail):
                notes.append("read its own response file before writing it")
            else:
                violations.append(f"read {target}")
        elif tool in WRITE_TOOLS:
            target, created = _edit_target(message)
            if message.get("tool_result") is None:
                notes.append(f"made a {tool} call that has no recorded result (a failed write, retried)")
                continue
            if target is not None and not _ends_with(target, response_tail):
                violations.append(f"wrote {target}")
                continue
            wrote_response = True
            written = created
        else:
            violations.append(f"used tool {tool}")
    return {"violations": violations, "notes": notes, "wrote_response": wrote_response, "written": written}


def _first_user_text(messages: list[dict]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("text") or "")
    return None


def _parse_time_ms(stamp: str) -> int:
    import datetime as dt

    return int(dt.datetime.fromisoformat(stamp).timestamp() * 1000)


def audit_session(session: Session, launches: list[Launch], transcripts: dict[str, dict], reused: set[str]) -> dict:
    state = session.load()
    bound = state["subagent_model"]
    run_dir = session.root.name
    started_ms = _parse_time_ms(state["created"])
    records = {record["candidate_id"]: record for record in read_records(session.root / "trajectory.jsonl")}
    calls: dict[str, list[dict]] = {}
    for call in read_records(session.root / "calls.jsonl"):
        calls.setdefault(call["candidate_id"], []).append(call)

    own = [launch for launch in launches if launch.cid and launch.cid.startswith(state["run_id"] + "-g")]
    own = [launch for launch in own if _ends_with(launch.prompt_path, f"{run_dir}/prompts/{launch.cid}.prompt.md")]
    before, during = [], []
    for launch in own:
        created = (transcripts.get(launch.agent_id) or {}).get("created_at_ms")
        (before if created is not None and created < started_ms else during).append(launch)
    by_cid: dict[str, list[Launch]] = {}
    for launch in sorted(during, key=lambda item: item.order):
        by_cid.setdefault(launch.cid, []).append(launch)

    rows = []
    for cid in sorted(set(records) | set(calls) | set(by_cid)):
        record = records.get(cid)
        cid_launches = by_cid.get(cid, [])
        cid_calls = calls.get(cid, [])
        prompt_tail = f"{run_dir}/prompts/{cid}.prompt.md"
        response_tail = f"{run_dir}/responses/{cid}.response.md"
        violations: list[str] = []
        notes: list[str] = []
        status = "pending" if record is None else ((record.get("eval") or {}).get("status") or "unknown")

        if len(cid_launches) != len(cid_calls):
            violations.append(f"{len(cid_launches)} launches but {len(cid_calls)} calls logged in calls.jsonl")
        for launch in cid_launches:
            if launch.model != bound:
                violations.append(f"launch {launch.agent_id} requested model {launch.model!r}, session is bound to {bound!r}")
            expected = subagent_instructions(launch.prompt_path or "", launch.response_path or "")
            if launch.instructions.strip() != expected.strip() or not _ends_with(launch.response_path, response_tail):
                violations.append(f"launch {launch.agent_id} did not use the standard instructions for {cid}")
            if launch.agent_id in reused:
                violations.append(f"agent {launch.agent_id} was launched for more than one candidate")

        response_file = session.root / "responses" / f"{cid}.response.md"
        if record:
            accepted = record.get("agent_id")
            expected_response = record.get("response") or ""
        else:
            logged_ok = [call.get("agent_id") for call in cid_calls if not call.get("error")]
            accepted = logged_ok[-1] if logged_ok else None
            expected_response = response_file.read_text() if response_file.exists() else ""
            if accepted:
                status = "awaiting_score"
        verified = False
        unverified: list[str] = []
        for launch in cid_launches:
            transcript = transcripts.get(launch.agent_id)
            if transcript is None:
                unverified.append(launch.agent_id)
                continue
            first = _first_user_text(transcript["messages"])
            if first is not None and first.strip() != launch.instructions.strip():
                violations.append(f"agent {launch.agent_id} received different instructions than the launch recorded")
            seen = inspect_subagent(transcript["messages"], prompt_tail, response_tail)
            violations.extend(f"agent {launch.agent_id} {item}" for item in seen["violations"])
            notes.extend(f"agent {launch.agent_id} {item}" for item in seen["notes"])
            if launch.agent_id == accepted:
                if not seen["wrote_response"]:
                    violations.append(f"accepted agent {accepted} never wrote the response file")
                elif seen["written"] is None:
                    notes.append(f"content written by {accepted} could not be reconstructed")
                elif _sha(seen["written"]) != _sha(expected_response):
                    what = "scored response" if record else "response file"
                    violations.append(f"{what} differs from what agent {accepted} wrote")
                else:
                    verified = True
            elif seen["wrote_response"]:
                violations.append(f"agent {launch.agent_id} wrote a response that was logged as failed or discarded")

        if record and status != "call_failed":
            if not accepted:
                violations.append("scored record names no agent")
            elif accepted not in {launch.agent_id for launch in cid_launches}:
                violations.append(f"scored record's agent {accepted} has no launch in the driver transcript")
            logged_ok = [call.get("agent_id") for call in cid_calls if not call.get("error")]
            if logged_ok and logged_ok[-1] != accepted:
                violations.append(f"calls.jsonl names {logged_ok[-1]} but the record names {accepted}")
        if unverified:
            notes.append(f"no subagent transcript for {', '.join(unverified)}")

        rows.append(
            {
                "candidate_id": cid,
                "status": status,
                "accepted_agent": accepted,
                "launches": [
                    {
                        "agent_id": launch.agent_id,
                        "model": launch.model,
                        "transcript": launch.agent_id in transcripts,
                        "error": next(
                            (call.get("error") for call in cid_calls if call.get("agent_id") == launch.agent_id),
                            None,
                        ),
                    }
                    for launch in cid_launches
                ],
                "logged_calls": len(cid_calls),
                "logged_failures": sum(1 for call in cid_calls if call.get("error")),
                "response_verified": verified,
                "violations": violations,
                "notes": notes,
            }
        )

    return {
        "run_id": state["run_id"],
        "bound_model": bound,
        "audited_at": _now(),
        "session_created": state["created"],
        "launches_before_session": [
            {"agent_id": launch.agent_id, "candidate_id": launch.cid, "model": launch.model} for launch in before
        ],
        "summary": {
            "candidates": len(rows),
            "launches": sum(len(row["launches"]) for row in rows),
            "responses_verified": sum(1 for row in rows if row["response_verified"]),
            "call_failed": sum(1 for row in rows if row["status"] == "call_failed"),
            "candidates_with_violations": sum(1 for row in rows if row["violations"]),
            "violations": sum(len(row["violations"]) for row in rows),
            "notes": sum(len(row["notes"]) for row in rows),
        },
        "candidates": rows,
        "_launches": own,
    }


def reused_agents(launches: list[Launch]) -> set[str]:
    seen: dict[str, int] = {}
    for launch in launches:
        seen[launch.agent_id] = seen.get(launch.agent_id, 0) + 1
    return {agent for agent, count in seen.items() if count > 1}


def archive(session: Session, result: dict, transcripts: dict[str, dict]) -> None:
    launches = result["_launches"]
    lines = "".join(json.dumps(asdict(launch), ensure_ascii=False) + "\n" for launch in launches)
    (session.root / "launches.jsonl").write_text(lines, encoding="utf-8")
    out = session.root / "agent_transcripts"
    out.mkdir(exist_ok=True)
    for launch in launches:
        entry = transcripts.get(launch.agent_id)
        if entry is not None:
            payload = {"agent_id": launch.agent_id, **entry}
            (out / f"{launch.agent_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )


def write_audit(session: Session, result: dict) -> Path:
    path = session.root / "audit.json"
    public = {key: value for key, value in result.items() if not key.startswith("_")}
    path.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_audit_report(root: Path, results: list[dict]) -> Path:
    lines = [
        "# Driver audit",
        "",
        "Each session's `calls.jsonl`, `meta/` and `trajectory.jsonl` checked against the platform transcripts "
        "of the driver and of every subagent it launched. Details are in each session's `audit.json`.",
        "",
        "| run | bound model | candidates | launches | responses verified | call_failed | violations | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            f"| {result['run_id']} | {result['bound_model']} | {summary['candidates']} | {summary['launches']} | "
            f"{summary['responses_verified']} | {summary['call_failed']} | {summary['violations']} | {summary['notes']} |"
        )
    details = []
    for result in results:
        for row in result["candidates"]:
            for item in row["violations"]:
                details.append(f"- violation, `{row['candidate_id']}`: {item}")
            for item in row["notes"]:
                details.append(f"- note, `{row['candidate_id']}`: {item}")
        for launch in result["launches_before_session"]:
            details.append(
                f"- note, `{launch['candidate_id']}`: agent {launch['agent_id']} was launched before this session "
                "was created (an earlier run with the same id) and is not counted"
            )
    if details:
        lines += ["", "## Findings", "", *details]
    path = root / "AUDIT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
