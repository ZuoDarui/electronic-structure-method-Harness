import json

import pytest
import yaml

from esm_harness import loop
from esm_harness.cli import main
from esm_harness.llm.client import StubClient
from esm_harness.search.session import Session, answer_with_client
from tests.test_search import LEGAL, _config


def _setup(tmp_path, generations=2):
    (tmp_path / "esm_harness").mkdir()
    (tmp_path / "esm_harness" / "core.py").write_text("x = 1\n", encoding="utf-8")
    config = _config(
        tmp_path, max_generations=generations, out_dir=str(tmp_path / "trajectories"), allow_any_model=True
    )
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {
        "rounds_dir": "trajectories/rounds",
        "sessions": [{"config": str(tmp_path / "run.yaml"), "run_id": "loop-test"}],
    }
    (tmp_path / "loop.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    return plan, Session(tmp_path / "trajectories" / "loop-test")


def _answer(session):
    answer_with_client(session, StubClient([LEGAL]))


def _tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_one_round_per_driver(tmp_path):
    plan, session = _setup(tmp_path)
    opened = loop.begin(tmp_path, plan, "bc-round-1")
    assert opened["round"] == 1 and [r["candidate_id"] for r in opened["requests"]] == ["loop-test-g00-s0"]
    assert "Use the Read tool once" in opened["requests"][0]["instructions"]
    with pytest.raises(loop.LoopError, match="held by bc-round-1"):
        loop.begin(tmp_path, plan, "bc-intruder")
    assert loop.begin(tmp_path, plan, "bc-round-1")["resumed"] is True
    with pytest.raises(loop.LoopError, match="still open"):
        loop.end(tmp_path, plan, "bc-round-1")
    _answer(session)
    with pytest.raises(loop.LoopError, match="belongs to bc-round-1"):
        loop.end(tmp_path, plan, "bc-intruder")
    report = loop.end(tmp_path, plan, "bc-round-1")
    assert loop.read_lock(tmp_path) is None
    [item] = report["sessions"]
    assert item["generation_scored"] == 0 and item["next_generation"] == 1
    assert item["candidates"][0]["status"] == "ok"
    assert (tmp_path / "trajectories" / "rounds" / "round-001.json").is_file()

    second = loop.begin(tmp_path, plan, "bc-round-2")
    assert second["round"] == 2 and second["requests"][0]["candidate_id"] == "loop-test-g01-s0"
    _answer(session)
    assert loop.end(tmp_path, plan, "bc-round-2")["sessions"][0]["status"] == "done"
    with pytest.raises(loop.LoopError, match="nothing to run"):
        loop.begin(tmp_path, plan, "bc-round-3")


def test_harness_edits_during_a_round_block_scoring(tmp_path):
    plan, session = _setup(tmp_path)
    loop.begin(tmp_path, plan, "bc-round-1")
    _answer(session)
    core = tmp_path / "esm_harness" / "core.py"
    core.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(loop.LoopError, match="harness files changed"):
        loop.end(tmp_path, plan, "bc-round-1")
    assert session.load()["generation"] == 0
    core.write_text("x = 1\n", encoding="utf-8")
    loop.end(tmp_path, plan, "bc-round-1")
    assert session.load()["generation"] == 1


def test_monitor_only_reads(tmp_path, capsys):
    plan, session = _setup(tmp_path)
    loop.begin(tmp_path, plan, "bc-round-1")
    before = _tree(tmp_path)
    view = loop.snapshot(tmp_path, plan)
    text = loop.format_snapshot(view)
    with pytest.raises(SystemExit) as stop:
        main(["monitor", "--root", str(tmp_path), "--json"])
    assert stop.value.code == 0
    assert _tree(tmp_path) == before
    assert view["state"] == "active" and view["lock"]["harness_unchanged"] is True
    assert view["sessions"][0]["pending"][0]["response"] is False
    assert "ACTIVE, round 1" in text
    assert json.loads(capsys.readouterr().out)["lock"]["driver_agent_id"] == "bc-round-1"


def test_loop_state_exit_codes(tmp_path):
    plan, session = _setup(tmp_path)
    with pytest.raises(SystemExit) as idle:
        main(["loop-state", "--root", str(tmp_path)])
    assert idle.value.code == 0
    loop.begin(tmp_path, plan, "bc-round-1")
    with pytest.raises(SystemExit) as active:
        main(["loop-state", "--root", str(tmp_path)])
    assert active.value.code == 3
