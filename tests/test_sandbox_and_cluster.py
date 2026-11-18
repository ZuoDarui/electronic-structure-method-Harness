import json
import os
import stat
import textwrap
from pathlib import Path

from esm_harness.config import REPO_ROOT
from esm_harness.eval.cases import load_cases
from esm_harness.eval.sandbox import evaluate_program
from esm_harness.eval.submit import SlurmSubmitter


H2 = {
    "id": "h2_sto3g",
    "atom": "H 0 0 0; H 0 0 0.74",
    "basis": "sto-3g",
    "charge": 0,
    "spin": 0,
    "unit": "Angstrom",
}

FACTORS = {
    "model": "claude-opus-5.5",
    "prompt_id": "hf_spec",
    "search": "iterative",
    "speed_weight": 0.3,
    "reasoning_effort": "high",
    "population_n": 1,
    "allowed_imports": ["numpy", "scipy", "pyscf.gto"],
    "max_abs_err": 1e-6,
    "max_time_ratio": 1.5,
    "fitness_err_ceiling": 1e-5,
}


def _install_fake_sbatch(directory: Path) -> None:
    """A stand-in scheduler: it accepts the batch script and runs that task."""
    script = directory / "sbatch"
    script.write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            batch=""
            for arg in "$@"; do
              case "$arg" in
                --wait) ;;
                *) batch="$arg" ;;
              esac
            done
            bash "$batch"
            echo "Submitted batch job 4242"
            """
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def test_cluster_submission_evaluates_the_job_script(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _install_fake_sbatch(bindir)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-reach-the-job")
    code = (REPO_ROOT / "tests" / "fixtures" / "legal_rhf.py").read_text(encoding="utf-8")
    references = {"h2_sto3g": -1.1167593073964255}
    submitter = SlurmSubmitter(REPO_ROOT, partition="cpu", account="esm")
    result = evaluate_program(
        code,
        cases=[H2],
        references=references,
        baseline_times={"h2_sto3g": 1.0},
        baseline_energies={"h2_sto3g": references["h2_sto3g"]},
        submitter=submitter,
        factors=FACTORS,
        timeout_s=120,
        memory_mb=8192,
        work_root=tmp_path / "jobs",
        repeats=1,
    )
    assert result["status"] == "ok", result["stderr"]
    assert result["submission"]["backend"] == "slurm"
    assert result["submission"]["scheduler_id"] == "4242"
    assert "#SBATCH --partition=cpu" in result["submission"]["script"]
    assert "#SBATCH --account=esm" in result["submission"]["script"]
    assert "*API_KEY*" in result["submission"]["script"]
    assert result["max_abs_err"] < 1e-8
    assert result["cases"][0]["converged"] is True
    job_dir = next((tmp_path / "jobs").iterdir())
    saved = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    script = (job_dir / "submit.sbatch").read_text(encoding="utf-8")
    assert "should-not-reach-the-job" not in script


def test_sandbox_rejects_without_submitting(tmp_path):
    calls = []

    class Boom:
        def submit(self, spec, job_file):
            calls.append(job_file)
            raise AssertionError("rejected source must not be submitted")

    result = evaluate_program(
        "from pyscf import scf\n\ndef solve(mol):\n    return scf.RHF(mol).kernel()\n",
        cases=[H2],
        references={"h2_sto3g": -1.0},
        baseline_times={"h2_sto3g": 1.0},
        baseline_energies={"h2_sto3g": -1.0},
        submitter=Boom(),
        factors=FACTORS,
        timeout_s=30,
        memory_mb=1024,
        work_root=tmp_path,
    )
    assert result["status"] == "rejected"
    assert calls == []
    assert result["fitness"] is None


def test_suite_cases_are_fixed():
    cases = load_cases(REPO_ROOT / "benchmarks" / "hf_rhf.yaml")
    assert [case["id"] for case in cases] == [
        "h2_sto3g",
        "heh_sto3g",
        "h2o_sto3g",
        "h2_631g",
        "h2o_631g",
        "nh3_631gs",
        "ch4_ccpvdz",
    ]


def test_child_environment_drops_credentials(monkeypatch, tmp_path):
    from esm_harness.eval.submit import JobSpec, _child_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("HOME_SAFE", "kept")
    spec = JobSpec(job_id="j", workdir=tmp_path, output_path=tmp_path / "o.json", timeout_s=1, env={"ESM_MEMORY_MB": "1"})
    env = _child_env(spec, REPO_ROOT)
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["HOME_SAFE"] == "kept"
    assert env["ESM_MEMORY_MB"] == "1"


def test_module_state_does_not_carry_between_timed_calls(tmp_path):
    from esm_harness.eval.submit import LocalSubmitter

    cheat = (
        "CACHE = {}\n"
        "def solve(mol):\n"
        "    if 'e' in CACHE:\n"
        "        return {'e_tot': CACHE['e'], 'converged': True}\n"
        "    CACHE['e'] = -9.0\n"
        "    return {'e_tot': -1.0, 'converged': True}\n"
    )
    result = evaluate_program(
        cheat,
        cases=[H2],
        references={"h2_sto3g": -1.0},
        baseline_times={"h2_sto3g": 1.0},
        baseline_energies={"h2_sto3g": -1.0},
        submitter=LocalSubmitter(REPO_ROOT),
        factors=FACTORS,
        timeout_s=60,
        memory_mb=4096,
        work_root=tmp_path,
        repeats=3,
    )
    assert result["status"] == "ok", result["stderr"]
    assert result["cases"][0]["energy"] == -1.0
