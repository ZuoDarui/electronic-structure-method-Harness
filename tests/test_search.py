import json

import pytest

from esm_harness.cli import main
from esm_harness.config import REPO_ROOT, cell_id, expand_grid, load_yaml, subagent_model
from esm_harness.eval.submit import make_submitter
from esm_harness.llm.client import StubClient, normalize_usage, split_program
from esm_harness.search.session import Session, run_with_client

LEGAL = (REPO_ROOT / "tests" / "fixtures" / "legal_rhf.py").read_text(encoding="utf-8")
BROKEN = "def solve(mol):\n    raise RuntimeError('not yet')\n"


def _config(tmp_path, **overrides):
    config = load_yaml(REPO_ROOT / "configs" / "hf_rhf.yaml")
    config["suite"] = str(tmp_path / "suite.yaml")
    (tmp_path / "suite.yaml").write_text(
        "cases:\n  - id: h2_sto3g\n    atom: H 0 0 0; H 0 0 0.74\n    basis: sto-3g\n",
        encoding="utf-8",
    )
    config["repeats"] = 1
    config["max_generations"] = 2
    config.update(overrides)
    return config


def test_split_program_and_usage():
    code, error = split_program("text\n```python\ndef solve(mol):\n    pass\n```\n")
    assert error == "" and code.startswith("def solve")
    assert split_program("no code here")[0] == ""
    usage = normalize_usage({"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 2})
    assert usage == {
        "input_tokens": 9,
        "cached_tokens": 2,
        "output_tokens": 4,
        "reasoning_tokens": 0,
        "total_tokens": 13,
    }
    assert normalize_usage(None) is None


def test_config_is_bound_to_opus():
    config = load_yaml(REPO_ROOT / "configs" / "hf_rhf.yaml")
    assert config["factors"]["model"] == "claude-opus-5.5"
    assert subagent_model(config["factors"]) == "claude-opus-5-5-high"
    with pytest.raises(ValueError):
        subagent_model({**config["factors"], "model": "grok-4.7"})


def test_ablation_cells_vary_one_factor():
    document = load_yaml(REPO_ROOT / "ablations" / "hf_rhf.yaml")
    cells = expand_grid(document)
    assert len(cells) == 8
    assert all(cell["factors"]["model"] == "claude-opus-5.5" for cell in cells)
    names = [cell_id(index, cell["factors"]) for index, cell in enumerate(cells)]
    assert len(set(names)) == len(names)
    assert {cell["factors"]["reasoning_effort"] for cell in cells} == {"medium", "high", "max"}


def test_generation_zero_prompt_has_no_program(tmp_path):
    config = _config(tmp_path)
    session = Session(tmp_path / "run")
    session.init(config, make_submitter(config, REPO_ROOT))
    [request] = session.requests()
    assert request["model"] == "claude-opus-5-5-high"
    prompt = open(request["prompt"], encoding="utf-8").read()
    assert "Define solve(mol)" in prompt
    assert "```python" not in prompt
    assert "Best program so far" not in prompt


def test_step_waits_then_scores_and_feeds_back(tmp_path):
    config = _config(tmp_path)
    submitter = make_submitter(config, REPO_ROOT)
    session = Session(tmp_path / "run")
    session.init(config, submitter)
    status = session.advance(submitter)
    assert status["candidates_scored"] == 0 and status["generation"] == 0

    [request] = session.requests()
    open(request["response"], "w", encoding="utf-8").write(f"```python\n{LEGAL}\n```\n")
    with pytest.raises(SystemExit) as caught:
        main(["meta", "--session", str(session.root), "--cid", request["candidate_id"],
              "--model", "claude-opus-5-5-high", "--agent-id", "agent-1"])
    assert caught.value.code == 0
    status = session.advance(submitter)
    assert status["generation"] == 1
    assert status["best"]["max_abs_err"] < 1e-8
    [request] = session.requests()
    prompt = open(request["prompt"], encoding="utf-8").read()
    assert "Best program so far" in prompt and "signed_err" in prompt

    record = json.loads(session.trajectory.read_text(encoding="utf-8").splitlines()[0])
    assert record["from_scratch"] is True
    assert record["model"] == "claude-opus-5-5-high"
    assert record["agent_id"] == "agent-1"
    assert record["usage"] is None
    assert record["eval"]["cases"][0]["abs_err_baseline"] < 1e-8
    assert record["eval"]["cases"][0]["time_ratio"] > 0


def test_session_refuses_another_model(tmp_path):
    config = _config(tmp_path)
    submitter = make_submitter(config, REPO_ROOT)
    session = Session(tmp_path / "run")
    session.init(config, submitter)
    [request] = session.requests()
    open(request["response"], "w", encoding="utf-8").write(f"```python\n{LEGAL}\n```\n")
    open(request["meta"], "w", encoding="utf-8").write(json.dumps({"model": "grok-4.7"}))
    with pytest.raises(ValueError, match="bound to"):
        session.advance(submitter)


def test_client_run_writes_ledger(tmp_path):
    config = _config(tmp_path, allow_any_model=True, max_generations=3)
    client = StubClient([BROKEN, LEGAL, LEGAL])
    ledger = run_with_client(config, client, make_submitter(config, REPO_ROOT), tmp_path / "run")
    assert ledger["status"] == "done"
    assert ledger["calls"] == 3
    assert ledger["usage"]["total_tokens"] == 151 + 152 + 153
    assert ledger["best"]["generation"] in (1, 2)
    assert ledger["generations"][0]["best"] is None
    assert "Most recent attempt" not in client.prompts[0]
    assert "not yet" in client.prompts[1]


def test_best_of_n_writes_one_prompt_per_sample(tmp_path):
    config = _config(tmp_path)
    config["factors"] = {**config["factors"], "search": "best_of_n", "population_n": 3}
    session = Session(tmp_path / "run")
    session.init(config, make_submitter(config, REPO_ROOT))
    assert [item["candidate_id"][-3:] for item in session.requests()] == ["-s0", "-s1", "-s2"]


def test_repeated_call_failures_are_scored_and_the_search_moves_on(tmp_path):
    config = _config(tmp_path, max_call_failures=2)
    submitter = make_submitter(config, REPO_ROOT)
    session = Session(tmp_path / "run")
    session.init(config, submitter)
    [request] = session.requests()
    for _ in range(2):
        with pytest.raises(SystemExit):
            main(["meta", "--session", str(session.root), "--cid", request["candidate_id"],
                  "--model", "claude-opus-5-5-high", "--error", "Activity task timed out"])
    assert session.requests() == []
    status = session.advance(submitter)
    assert status["generation"] == 1
    record = json.loads(session.trajectory.read_text(encoding="utf-8").splitlines()[0])
    assert record["eval"]["status"] == "call_failed"
    assert record["call_failures"] == 2
    assert record["eval"]["fitness"] is None


def test_rescore_retimes_recorded_programs_against_pyscf(tmp_path):
    from esm_harness.rescore import rescore_session

    config = _config(tmp_path, allow_any_model=True, max_generations=1)
    submitter = make_submitter(config, REPO_ROOT)
    run_with_client(config, StubClient([LEGAL]), submitter, tmp_path / "run")
    before = (tmp_path / "run" / "trajectory.jsonl").read_text(encoding="utf-8")
    result = rescore_session(Session(tmp_path / "run"), REPO_ROOT / "baselines" / "pyscf_rhf.py", submitter)
    assert (tmp_path / "run" / "trajectory.jsonl").read_text(encoding="utf-8") == before
    [row] = result["rows"]
    assert row["status"] == "ok"
    assert row["cases"][0]["abs_err_baseline"] < 1e-8
    assert row["cases"][0]["baseline_time_s"] == result["baseline_times"]["h2_sto3g"]
    assert (tmp_path / "run" / "rescore" / "pyscf_rhf.jsonl").is_file()
