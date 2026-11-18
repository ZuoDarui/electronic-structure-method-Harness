import json
import shlex
from pathlib import Path

import pytest
import yaml

from esm_harness import loop
from esm_harness.config import REPO_ROOT, load_yaml
from esm_harness.eval.cases import load_cases
from esm_harness.eval.policy import rejection_reason
from esm_harness.eval.reference import read_reference_records, select_reference
from esm_harness.eval.submit import JobSpec, LocalSubmitter, TioneSubmitter
from esm_harness.llm.client import StubClient
from esm_harness.llm.prompts import render_prompt
from esm_harness.search.session import Session, answer_with_client

SUITE = REPO_ROOT / "benchmarks" / "hf_rijcosx_cluster.yaml"
REFERENCES = REPO_ROOT / "benchmarks" / "rijcosx_cluster_references.json"
CONFIG = REPO_ROOT / "configs" / "hf_rijcosx_cluster.yaml"


def test_suite_is_the_cluster_water_and_alkane_set():
    cases = load_cases(SUITE)
    assert [case["id"] for case in cases] == [
        "water8", "water16", "water32", "water64", "water128",
        "c21h44", "c43h88", "c86h174", "c172h346",
    ]
    by_id = {case["id"]: case for case in cases}
    assert len(by_id["water8"]["atom"].split(";")) == 24
    assert len(by_id["c172h346"]["atom"].split(";")) == 518
    assert all(case["basis"] == "cc-pvtz" for case in cases)
    assert by_id["water8"]["method"] == {
        "jk": "rijcosx", "auxbasis": "weigend", "grid": [30, 110], "final_grid": [35, 302],
        "conv_tol": 1e-8, "final_energy": "single_point",
    }
    assert by_id["c172h346"]["method"]["final_energy"] == "reconverge"
    assert by_id["c172h346"]["method"]["grid"] == [30, 110]


def test_suite_basis_sizes_match_the_cluster_runs():
    from pyscf import gto

    for case in load_cases(SUITE):
        mol = gto.M(atom=case["atom"], basis=case["basis"], verbose=0)
        assert mol.nao_nr() == case["nao"], case["id"]


def test_reference_is_pyscf_then_hf_py():
    records = read_reference_records(REFERENCES, load_cases(SUITE))
    sources = {case_id: record["source"] for case_id, record in records.items()}
    assert sources == {
        "water8": "pyscf", "water16": "pyscf", "water32": "pyscf", "water64": "pyscf",
        "water128": "hf.py", "c21h44": "pyscf", "c43h88": "pyscf", "c86h174": "pyscf",
        "c172h346": "hf.py",
    }
    assert records["water8"]["energy"] == -608.450816454237
    assert records["c172h346"]["energy"] == -6717.379184800415
    assert all(record["time_s"] > 0 for record in records.values())
    assert select_reference("x", -1.5) == {"energy": -1.5, "source": None, "time_s": None}
    with pytest.raises(ValueError, match="no converged reference"):
        select_reference("x", {"pyscf": {"energy": None, "converged": False}, "hf.py": {"converged": False}})


def test_rijcosx_prompt_states_method_imports_and_floor():
    factors = load_yaml(CONFIG)["factors"]
    prompt = render_prompt("hf_rijcosx", load_cases(SUITE), None, None, allowed_imports=factors["allowed_imports"])
    assert "solve(mol, method)" in prompt
    assert "pyscf.dft.gen_grid" in prompt and "treutler_ahlrichs" in prompt
    assert "within 1e-8" in prompt and "never used as the best program" in prompt
    assert "- c172h346: C172H346, 518 atoms, basis cc-pvtz, 10004 basis functions, final_energy reconverge" in prompt
    assert "0.0000000000" not in prompt


def test_policy_allows_only_the_listed_grid_modules():
    allowed = load_yaml(CONFIG)["factors"]["allowed_imports"]
    body = "def solve(mol, method):\n    return {'e_tot': -1.0, 'converged': True}\n"
    assert rejection_reason("from pyscf.dft import gen_grid, radi\nimport pyscf.dft.gen_grid\n" + body, allowed) is None
    assert rejection_reason("from pyscf.dft import rks\n" + body, allowed)
    assert rejection_reason("from pyscf import dft\n" + body, allowed)
    assert rejection_reason("from pyscf import sgx\n" + body, allowed)
    assert "pyscf.dft drivers" in rejection_reason("from pyscf.dft import gen_grid\nx = gen_grid.dft.RKS\n" + body, allowed)


# A program that knows the two test molecules apart by bond length.
EXACT = """\
def solve(mol, method):
    assert method["final_energy"] == "single_point"
    z = float(mol.atom_coords()[1][2])
    return {"e_tot": -1.0 if z < 1.45 else -2.0, "converged": True}
"""
OFF_BY_5E8 = EXACT.replace("-2.0", "-2.00000005")


class FakeTione:
    """Runs each job locally at start, but reports it finished only after ``finish()``."""

    backend = "tione"
    asynchronous = True

    def __init__(self):
        self.local = LocalSubmitter(REPO_ROOT)
        self.finished = set()
        self.started = []

    def preflight(self):
        return None

    def finish(self):
        self.finished.update(handle["task_id"] for handle in self.started)

    def start(self, spec, job_file, label=""):
        submission = self.local.submit(spec, job_file)
        handle = {"output": str(spec.output_path), "task_id": f"t{len(self.started)}", "label": label,
                  "returncode": submission.returncode}
        self.started.append(handle)
        return handle

    def poll(self, handle):
        if handle["task_id"] not in self.finished:
            return None
        payload = json.loads(Path(handle["output"]).read_text(encoding="utf-8"))
        return {"payload": payload, "submission": {"backend": "tione", "scheduler_id": handle["task_id"], "returncode": 0}}


def _cluster_config(tmp_path, **overrides):
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "method": {"jk": "rijcosx", "final_energy": "single_point"},
                "cases": [
                    {"id": "h2a", "atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"},
                    {"id": "h2b", "atom": "H 0 0 0; H 0 0 0.80", "basis": "sto-3g"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "refs.json").write_text(
        json.dumps(
            {
                "h2a": {"pyscf": {"energy": -1.0, "converged": True, "time_s": 2.0},
                        "hf.py": {"energy": -1.1, "converged": True, "time_s": 1.0}},
                "h2b": {"pyscf": {"energy": None, "converged": False, "time_s": None},
                        "hf.py": {"energy": -2.0, "converged": True, "time_s": 4.0}},
            }
        ),
        encoding="utf-8",
    )
    config = load_yaml(CONFIG)
    config.update(
        suite=str(tmp_path / "suite.yaml"),
        references=str(tmp_path / "refs.json"),
        out_dir=str(tmp_path / "trajectories"),
        allow_any_model=True,
        max_generations=3,
        patience=None,
    )
    config.update(overrides)
    return config


def test_cluster_session_scores_only_after_every_job_finishes(tmp_path):
    submitter = FakeTione()
    session = Session(tmp_path / "run")
    session.init(_cluster_config(tmp_path), submitter)
    baseline = session.baseline()
    assert baseline["host"] == "recorded"
    assert baseline["times"] == {"h2a": 2.0, "h2b": 4.0}
    assert baseline["time_sources"] == {"h2a": "pyscf", "h2b": "hf.py"}
    assert session.remeasure_baseline(submitter) == baseline

    answer_with_client(session, StubClient([OFF_BY_5E8]))
    status = session.advance(submitter)
    assert status["status"] == "evaluating" and status["evaluating"] == ["hf-opus55-cluster-g00-s0"]
    assert status["candidates_scored"] == 0 and status["requests"] == []
    assert len(submitter.started) == 2
    assert all(handle["label"].startswith("hf-opus55-cluster-g00-s0-") for handle in submitter.started)
    assert session.advance(submitter)["status"] == "evaluating"
    assert len(submitter.started) == 2

    submitter.finish()
    status = session.advance(submitter)
    assert status["generation"] == 1 and status["status"] == "awaiting_responses"
    assert len(submitter.started) == 2
    [record] = [json.loads(line) for line in session.trajectory.read_text(encoding="utf-8").splitlines()]
    assert record["eval"]["status"] == "ok"
    assert record["eval"]["accepted"] is False and record["eval"]["fitness"] is None
    assert record["eval"]["max_abs_err"] == pytest.approx(5e-8, rel=1e-3)
    assert status["best"] is None
    prompt = session.path("prompts", "hf-opus55-cluster-g01-s0").read_text(encoding="utf-8")
    assert "accepted (every molecule converged and within 1e-8 Hartree): False" in prompt

    answer_with_client(session, StubClient([EXACT]))
    assert session.advance(submitter)["status"] == "evaluating"
    submitter.finish()
    status = session.advance(submitter)
    assert status["best"]["candidate_id"] == "hf-opus55-cluster-g01-s0"
    assert status["best"]["max_abs_err"] == 0.0
    record = [json.loads(line) for line in session.trajectory.read_text(encoding="utf-8").splitlines()][-1]
    assert record["eval"]["accepted"] is True
    assert [case["time_ratio"] is not None for case in record["eval"]["cases"]] == [True, True]


def test_round_waits_for_the_cluster_without_taking_the_lock(tmp_path, monkeypatch):
    config = _cluster_config(tmp_path)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {"rounds_dir": "trajectories/rounds", "sessions": [{"config": str(tmp_path / "run.yaml"), "run_id": "c"}]}
    submitter = FakeTione()
    monkeypatch.setattr(loop, "make_submitter", lambda config, root: submitter)
    session = Session(tmp_path / "trajectories" / "c")

    opened = loop.begin(tmp_path, plan, "bc-1")
    assert [r["candidate_id"] for r in opened["requests"]] == ["c-g00-s0"]
    answer_with_client(session, StubClient([EXACT]))
    report = loop.end(tmp_path, plan, "bc-1")
    [item] = report["sessions"]
    assert item["status"] == "evaluating" and item["next_generation"] is None
    assert [task["case_id"] for task in item["evaluating"][0]["tasks"]] == ["h2a", "h2b"]
    assert loop.read_lock(tmp_path) is None
    assert "evaluating on the cluster" in loop.format_snapshot(loop.snapshot(tmp_path, plan))

    with pytest.raises(loop.LoopError, match="waiting for cluster evaluations"):
        loop.begin(tmp_path, plan, "bc-2")
    assert loop.read_lock(tmp_path) is None

    submitter.finish()
    opened = loop.begin(tmp_path, plan, "bc-3")
    assert [r["candidate_id"] for r in opened["requests"]] == ["c-g01-s0"]
    assert session.status()["best"]["candidate_id"] == "c-g00-s0"


def test_round_end_reaches_the_cluster_with_the_current_config(tmp_path, monkeypatch):
    config = _cluster_config(tmp_path)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {"rounds_dir": "trajectories/rounds", "sessions": [{"config": str(tmp_path / "run.yaml"), "run_id": "c"}]}
    submitter = FakeTione()
    seen = []
    monkeypatch.setattr(loop, "make_submitter", lambda config, root: seen.append(config) or submitter)
    session = Session(tmp_path / "trajectories" / "c")
    loop.begin(tmp_path, plan, "bc-1")
    answer_with_client(session, StubClient([EXACT]))
    import shutil
    shutil.rmtree(session.root / "programs")
    config["cluster"] = {**config["cluster"], "remote_root": "/cfs/moved"}
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    loop.end(tmp_path, plan, "bc-1")
    assert seen[-1]["cluster"]["remote_root"] == "/cfs/moved"
    assert seen[-1]["factors"] == session.load()["config"]["factors"]
    assert session.path("programs", "c-g00-s0").is_file()


def test_round_stops_before_model_calls_when_the_cluster_is_unreachable(tmp_path, monkeypatch):
    config = _cluster_config(tmp_path)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {"rounds_dir": "trajectories/rounds", "sessions": [{"config": str(tmp_path / "run.yaml"), "run_id": "c"}]}
    monkeypatch.delenv("ESM_CLUSTER_PASSWORD", raising=False)
    monkeypatch.delenv("ESM_CLUSTER_KEY_FILE", raising=False)
    with pytest.raises(loop.LoopError, match="ESM_CLUSTER_PASSWORD"):
        loop.begin(tmp_path, plan, "bc-1")
    assert loop.read_lock(tmp_path) is None
    assert not (tmp_path / "trajectories" / "c").exists()


class FakeRemote:
    def __init__(self, submit_failures=0):
        self.files = {}
        self.commands = []
        self.status = "RUNNING"
        self.submit_failures = submit_failures

    def run(self, command, timeout=600):
        self.commands.append(command)
        if "esm_tione_submit.py" in command:
            if self.submit_failures:
                self.submit_failures -= 1
                return 1, "", "warnings\nsubmit=failed\nreason=no training task in this resource group has a VPC and subnet\n"
            return 0, 'name_ok=true\n{"submit": "created", "task_id": "train-123", "name": "x"}\n', ""
        if "tione_status.py" in command:
            return 0, json.dumps({"train-123": {"status": self.status, "reason": "oom"}}) + "\n", ""
        return 0, "", ""

    def put_text(self, path, text, mode=None):
        self.files[path] = text

    def read_text(self, path):
        return self.files.get(path)


def _tione(tmp_path, remote):
    cluster = load_yaml(CONFIG)["cluster"]
    submitter = TioneSubmitter(REPO_ROOT, {**cluster, "remote_root": "/cfs/eval"}, remote=remote)
    workdir = tmp_path / "job1"
    workdir.mkdir(parents=True)
    (workdir / "candidate.py").write_text(EXACT, encoding="utf-8")
    job_file = workdir / "job.json"
    job_file.write_text(json.dumps({"program": str(workdir / "candidate.py"), "output": str(workdir / "result.json"),
                                    "cases": [{"id": "h2a"}], "repeats": 1, "warmup": False}), encoding="utf-8")
    spec = JobSpec(job_id="abc123", workdir=workdir, output_path=workdir / "result.json", timeout_s=900)
    return submitter, spec, job_file


def test_tione_submitter_deploys_submits_and_collects(tmp_path):
    remote = FakeRemote()
    submitter, spec, job_file = _tione(tmp_path, remote)
    handle = submitter.start(spec, job_file, label="cid-h2a")
    assert handle["task_id"] == "train-123" and "submit_error" not in handle
    job = json.loads(remote.files["/cfs/eval/abc123/job.json"])
    assert job["program"] == "candidate.py" and job["output"] == "result.json" and job["warmup"] is False
    assert remote.files["/cfs/eval/abc123/candidate.py"] == EXACT
    assert "def run_job" in remote.files["/cfs/eval/abc123/worker.py"]
    script = remote.files["/cfs/eval/abc123/esm-abc123.sh"]
    assert "timeout --kill-after=60 900 python -u worker.py --job job.json" in script
    assert "miniforge3/bin/activate pyscf" in script and "OMP_NUM_THREADS=8" in script
    assert "*PASSWORD*" in script
    [submit] = [command for command in remote.commands if "esm_tione_submit.py" in command]
    args = shlex.split(shlex.split(submit)[2])
    assert args[:8] == ["uv", "run", "-q", "--with", "tikit", "python", "/cfs/eval/esm_tione_submit.py",
                        "~/.cursor/skills/server-use/scripts/submit_training_task.py"]
    options = dict(zip(args[8:-1:2], args[9:-1:2]))
    assert options["--host"] == "infra" and options["--name"].startswith("zuodarui-esm-abc123-")
    assert (options["--gpus"], options["--cpu"], options["--memory"], options["--image-type"]) == ("1", "48", "256", "CCR")
    assert options["--start-cmd"] == "bash /cfs/eval/abc123/esm-abc123.sh" and args[-1] == "--submit"
    assert "s.resolve_host = resolve_host" in remote.files["/cfs/eval/esm_tione_submit.py"]

    assert submitter.poll(handle) is None
    remote.files["/cfs/eval/abc123/exit_code"] = "0\n"
    remote.files["/cfs/eval/abc123/result.json"] = json.dumps({"status": "ok", "stderr": "", "cases": [{"id": "h2a"}]})
    result = submitter.poll(handle)
    assert result["payload"]["status"] == "ok"
    assert result["submission"]["scheduler_id"] == "train-123" and result["submission"]["returncode"] == 0


def test_tione_submitter_reports_timeouts_and_dead_tasks(tmp_path):
    remote = FakeRemote()
    submitter, spec, job_file = _tione(tmp_path, remote)
    handle = submitter.start(spec, job_file)
    remote.files["/cfs/eval/abc123/exit_code"] = "124\n"
    assert submitter.poll(handle)["payload"]["status"] == "timeout"

    del remote.files["/cfs/eval/abc123/exit_code"]
    remote.status = "FAILED"
    result = submitter.poll(handle)
    assert result["payload"]["status"] == "error" and "FAILED" in result["payload"]["stderr"]


def test_failed_task_creation_is_retried_not_scored(tmp_path, monkeypatch):
    from esm_harness.eval import submit as submit_module

    remote = FakeRemote(submit_failures=2)
    submitter, spec, job_file = _tione(tmp_path, remote)
    handle = submitter.start(spec, job_file)
    assert handle["task_id"] is None and "VPC and subnet" in handle["submit_error"]
    assert submitter.poll(handle) is None
    assert handle["task_id"] is None and handle["submit_attempts"] == 2
    assert submitter.poll(handle) is None
    assert handle["task_id"] == "train-123" and "submit_error" not in handle

    remote = FakeRemote(submit_failures=10)
    submitter, spec, job_file = _tione(tmp_path / "late", remote)
    handle = submitter.start(spec, job_file)
    monkeypatch.setattr(submit_module, "_now_s", lambda: handle["first_attempt"] + 259201)
    result = submitter.poll(handle)
    assert result["payload"]["status"] == "submit_failed" and "2 attempts" in result["payload"]["stderr"]


def test_abort_closes_the_round_without_scoring(tmp_path, monkeypatch):
    config = _cluster_config(tmp_path)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {"rounds_dir": "trajectories/rounds", "sessions": [{"config": str(tmp_path / "run.yaml"), "run_id": "c"}]}
    submitter = FakeTione()
    monkeypatch.setattr(loop, "make_submitter", lambda config, root: submitter)
    session = Session(tmp_path / "trajectories" / "c")
    loop.begin(tmp_path, plan, "bc-1")
    answer_with_client(session, StubClient([EXACT]))
    with pytest.raises(loop.LoopError, match="belongs to bc-1"):
        loop.abort(tmp_path, plan, "bc-2", "x")
    report = loop.abort(tmp_path, plan, "bc-1", "cluster could not create tasks")
    assert report["aborted"] == "cluster could not create tasks" and loop.read_lock(tmp_path) is None
    assert session.load()["generation"] == 0 and not session.trajectory.exists()
    assert "aborted: cluster could not create tasks" in loop.format_snapshot(loop.snapshot(tmp_path, plan))

    opened = loop.begin(tmp_path, plan, "bc-3")
    assert opened["round"] == 2 and opened["requests"] == []
    assert loop.end(tmp_path, plan, "bc-3")["sessions"][0]["status"] == "evaluating"


def test_worker_passes_the_suite_method_to_solve(tmp_path):
    from esm_harness.eval.sandbox import evaluate_program

    factors = load_yaml(CONFIG)["factors"]
    case = {"id": "h2a", "atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", "charge": 0, "spin": 0,
            "unit": "Angstrom", "method": {"final_energy": "single_point"}}
    result = evaluate_program(
        EXACT, cases=[case], references={"h2a": -1.0}, baseline_times={"h2a": 1.0},
        baseline_energies={"h2a": -1.0}, submitter=LocalSubmitter(REPO_ROOT), factors=factors,
        timeout_s=60, memory_mb=4096, work_root=tmp_path, repeats=1, warmup=False,
    )
    assert result["status"] == "ok", result["stderr"]
    assert result["accepted"] is True and result["cases"][0]["abs_err"] == 0.0
