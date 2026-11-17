import json

import pytest

from esm_harness.audit import audit_session, load_agent_transcripts, load_launches, reused_agents
from esm_harness.cli import main
from esm_harness.config import REPO_ROOT
from esm_harness.eval.submit import make_submitter
from esm_harness.search.session import Session
from tests.test_search import LEGAL, _config

RESPONSE = f"```python\n{LEGAL}```\n"


def _cli(*argv):
    with pytest.raises(SystemExit) as stop:
        main(list(argv))
    assert stop.value.code == 0


def _child(instructions, prompt, response, text=RESPONSE, extra=()):
    lines = text.splitlines()
    diff = f"--- /dev/null\n+++ b/{response}\n@@ -1,0 +1,{len(lines)} @@\n" + "\n".join("+" + line for line in lines)
    return [
        {"role": "user", "text": instructions},
        {"role": "tool", "tool_name": "read_file", "tool_args": {"target_file": prompt}, "tool_result": {}},
        *extra,
        {
            "role": "tool",
            "tool_name": "search_replace",
            "tool_result": {"resultType": "editFileResult", "value": {"diff": {"chunks": [{"diffString": diff}]}}},
        },
        {"role": "assistant", "text": "done"},
    ]


def _launch(agent, instructions, model):
    return {
        "role": "tool",
        "tool_name": "task_v2",
        "tool_args": {"prompt": instructions, "model": model},
        "tool_result": {"resultType": "taskV2Result", "value": {"agentId": agent, "cloudAgentBcId": agent}},
    }


def _drive(tmp_path, *, failures=0, failed_wrote=False, model=None, extra=(), reuse=False):
    """Answer one generation the way a Cursor driver does, recording transcripts as the platform would."""
    config = _config(tmp_path, max_generations=1)
    submitter = make_submitter(config, REPO_ROOT)
    session = Session(tmp_path / "run")
    session.init(config, submitter)
    bound = session.load()["subagent_model"]
    driver, children = [], {}
    [request] = session.requests()
    cid = request["candidate_id"]
    for attempt in range(failures):
        agent = f"bc-fail-{attempt}"
        driver.append(_launch(agent, request["instructions"], bound))
        children[agent] = (
            _child(request["instructions"], request["prompt"], request["response"]) if failed_wrote else []
        )
        _cli("meta", "--session", str(session.root), "--cid", cid, "--model", bound, "--error", "timed out")
    agent = "bc-fail-0" if reuse else "bc-ok"
    driver.append(_launch(agent, request["instructions"], model or bound))
    children[agent] = _child(request["instructions"], request["prompt"], request["response"], extra=extra)
    with open(request["response"], "w", encoding="utf-8") as handle:
        handle.write(RESPONSE)
    _cli("meta", "--session", str(session.root), "--cid", cid, "--model", bound, "--agent-id", agent)
    session.advance(submitter)

    (tmp_path / "driver.json").write_text(json.dumps({"messages": driver}), encoding="utf-8")
    for agent, messages in children.items():
        (tmp_path / "fetched" / agent).mkdir(parents=True, exist_ok=True)
        (tmp_path / "fetched" / agent / "transcript.json").write_text(json.dumps({"messages": messages}))
    return session, cid


def _audit(tmp_path, session):
    launches = load_launches([tmp_path / "driver.json"])
    transcripts = load_agent_transcripts([tmp_path / "fetched"])
    return audit_session(session, launches, transcripts, reused_agents(launches))


def test_faithful_driver_passes_and_response_is_verified(tmp_path):
    session, _ = _drive(tmp_path, failures=1)
    result = _audit(tmp_path, session)
    assert result["summary"]["violations"] == 0, result["candidates"]
    assert result["summary"]["responses_verified"] == 1
    assert result["summary"]["launches"] == 2


def test_edited_response_is_caught(tmp_path):
    session, _ = _drive(tmp_path)
    path = session.root / "trajectory.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["response"] = record["response"].replace("def solve", "def solve  ", 1)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("differs from what agent" in item for item in row["violations"])


def test_other_model_is_caught(tmp_path):
    session, _ = _drive(tmp_path, model="gpt-5.6-sol-high")
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("requested model 'gpt-5.6-sol-high'" in item for item in row["violations"])


def test_unlogged_launch_is_caught(tmp_path):
    session, cid = _drive(tmp_path)
    calls = session.root / "calls.jsonl"
    calls.write_text("", encoding="utf-8")
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("1 launches but 0 calls" in item for item in row["violations"])


def test_discarded_response_is_caught(tmp_path):
    session, _ = _drive(tmp_path, failures=1, failed_wrote=True)
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("logged as failed or discarded" in item for item in row["violations"])


def test_reused_agent_is_caught(tmp_path):
    session, _ = _drive(tmp_path, failures=1, reuse=True)
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("more than one candidate" in item for item in row["violations"])


def test_reading_other_files_is_caught(tmp_path):
    peek = {"role": "tool", "tool_name": "read_file", "tool_args": {"target_file": "/repo/esm_harness/eval/score.py"}}
    shell = {"role": "tool", "tool_name": "run_terminal_cmd", "tool_args": {"command": "ls"}}
    session, _ = _drive(tmp_path, extra=(peek, shell))
    [row] = _audit(tmp_path, session)["candidates"]
    assert any("read /repo/esm_harness/eval/score.py" in item for item in row["violations"])
    assert any("used tool run_terminal_cmd" in item for item in row["violations"])


def test_audit_cli_archives_and_reruns_from_the_session(tmp_path):
    session, _ = _drive(tmp_path)
    root = str(tmp_path)
    _cli("audit", "--root", root, "--driver-transcript", str(tmp_path / "driver.json"),
         "--transcripts", str(tmp_path / "fetched"), "--archive")
    assert (session.root / "launches.jsonl").is_file()
    assert (session.root / "agent_transcripts" / "bc-ok.json").is_file()
    first = json.loads((session.root / "audit.json").read_text(encoding="utf-8"))["summary"]
    _cli("audit", "--root", root, "--strict")
    again = json.loads((session.root / "audit.json").read_text(encoding="utf-8"))["summary"]
    assert first == again and again["responses_verified"] == 1
