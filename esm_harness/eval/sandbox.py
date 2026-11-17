"""Prepare a generated program and evaluate it through the job submitter.

``evaluate_program`` runs every case in one job and waits for it. Submitters
that queue work elsewhere (``TioneSubmitter``) go through ``start_evaluation``,
which submits one job per case and returns at once, and ``collect_evaluation``,
which returns the evaluation once every job has finished.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from esm_harness.eval.policy import blocked_modules, rejection_reason
from esm_harness.eval.score import summarize_cases
from esm_harness.eval.submit import JobSpec

_EMPTY = {
    "max_abs_err": None,
    "max_abs_err_vs_baseline": None,
    "geomean_time_ratio": None,
    "all_converged": False,
    "accepted": False,
    "fitness": None,
}


def _rejected(code: str, factors: dict, trusted: bool) -> dict | None:
    reason = None if trusted else rejection_reason(code, list(factors["allowed_imports"]))
    if not reason:
        return None
    summary = summarize_cases([], factors["speed_weight"], factors["fitness_err_ceiling"])
    return {"status": "rejected", "stderr": reason, "cases": [], "submission": None, **summary}


def _write_job(
    code: str,
    cases: list[dict],
    *,
    factors: dict,
    timeout_s: float,
    memory_mb: int,
    work_root: Path,
    repeats: int,
    trusted: bool,
    warmup: bool,
) -> tuple[JobSpec, Path]:
    job_id = uuid.uuid4().hex[:12]
    workdir = work_root / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    program = workdir / "candidate.py"
    program.write_text(code, encoding="utf-8")
    output = workdir / "result.json"
    job = {
        "program": str(program),
        "output": str(output),
        "cases": cases,
        "repeats": repeats,
        "warmup": warmup,
        "memory_mb": memory_mb,
        "block_imports": [] if trusted else blocked_modules(list(factors["allowed_imports"])),
    }
    job_file = workdir / "job.json"
    job_file.write_text(json.dumps(job), encoding="utf-8")
    spec = JobSpec(
        job_id=job_id,
        workdir=workdir,
        output_path=output,
        timeout_s=timeout_s,
        env={"ESM_MEMORY_MB": str(memory_mb)},
    )
    return spec, job_file


def _submission_record(submission) -> dict:
    return {
        "backend": submission.backend,
        "scheduler_id": submission.scheduler_id,
        "returncode": submission.returncode,
        "script": submission.script,
    }


def _read_payload(spec: JobSpec, submission) -> dict:
    if spec.output_path.is_file():
        return json.loads(spec.output_path.read_text(encoding="utf-8"))
    if submission.returncode == 124:
        return {"status": "timeout", "stderr": submission.stderr[-16000:], "cases": []}
    return {
        "status": "error",
        "stderr": (submission.stderr or submission.stdout or "job wrote no result")[-16000:],
        "cases": [],
    }


def _join(
    payload: dict,
    submission,
    *,
    cases: list[dict],
    references: dict[str, float],
    baseline_times: dict[str, float],
    baseline_energies: dict[str, float],
    factors: dict,
) -> dict:
    joined = []
    for case in payload.get("cases") or []:
        ref = references.get(case["id"])
        base_energy = baseline_energies.get(case["id"])
        base_time = baseline_times.get(case["id"])
        energy = case.get("energy")
        ratio = None
        if base_time and case.get("time_s"):
            ratio = float(case["time_s"]) / float(base_time)
        joined.append(
            {
                "id": case["id"],
                "energy": energy,
                "ref_energy": ref,
                "baseline_energy": base_energy,
                "abs_err": None if energy is None or ref is None else abs(float(energy) - float(ref)),
                "abs_err_baseline": (
                    None
                    if energy is None or base_energy is None
                    else abs(float(energy) - float(base_energy))
                ),
                "converged": bool(case.get("converged")),
                "time_s": case.get("time_s"),
                "baseline_time_s": base_time,
                "time_ratio": ratio,
            }
        )
    status = payload.get("status", "error")
    if status == "ok" and len(joined) != len(cases):
        status = "error"
    summary = summarize_cases(
        joined if status == "ok" else [],
        factors["speed_weight"],
        factors["fitness_err_ceiling"],
    )
    return {
        "status": status,
        "stderr": payload.get("stderr") or "",
        "cases": joined,
        "submission": submission,
        **summary,
    }


def evaluate_program(
    code: str,
    *,
    cases: list[dict],
    references: dict[str, float],
    baseline_times: dict[str, float],
    baseline_energies: dict[str, float],
    submitter,
    factors: dict,
    timeout_s: float,
    memory_mb: int,
    work_root: Path,
    repeats: int = 3,
    trusted: bool = False,
    warmup: bool = True,
) -> dict:
    """``trusted`` is only for harness-owned code such as the baseline: it skips
    the generated-program policy and leaves the PySCF drivers importable."""
    rejected = _rejected(code, factors, trusted)
    if rejected:
        return rejected
    spec, job_file = _write_job(
        code, cases, factors=factors, timeout_s=timeout_s, memory_mb=memory_mb,
        work_root=work_root, repeats=repeats, trusted=trusted, warmup=warmup,
    )
    submission = submitter.submit(spec, job_file)
    return _join(
        _read_payload(spec, submission),
        _submission_record(submission),
        cases=cases,
        references=references,
        baseline_times=baseline_times,
        baseline_energies=baseline_energies,
        factors=factors,
    )


def start_evaluation(
    code: str,
    *,
    cases: list[dict],
    submitter,
    factors: dict,
    timeouts: dict[str, float],
    memory_mb: int,
    work_root: Path,
    repeats: int = 1,
    label: str = "",
) -> dict:
    """Submit one job per case. Returns ``{"evaluation": ...}`` when the program is
    rejected before running, otherwise ``{"jobs": [...]}`` to pass to ``collect_evaluation``."""
    rejected = _rejected(code, factors, False)
    if rejected:
        return {"evaluation": rejected}
    jobs = []
    for case in cases:
        spec, job_file = _write_job(
            code, [case], factors=factors, timeout_s=float(timeouts[case["id"]]), memory_mb=memory_mb,
            work_root=work_root, repeats=repeats, trusted=False, warmup=False,
        )
        handle = submitter.start(spec, job_file, label=f"{label}-{case['id']}" if label else case["id"])
        jobs.append({"case_id": case["id"], "job_id": spec.job_id, "handle": handle})
    return {"jobs": jobs}


def collect_evaluation(
    started: dict,
    *,
    submitter,
    cases: list[dict],
    references: dict[str, float],
    baseline_times: dict[str, float],
    baseline_energies: dict[str, float],
    factors: dict,
) -> dict | None:
    """The evaluation once every job of ``started`` has finished, otherwise None."""
    if "evaluation" in started:
        return started["evaluation"]
    finished = []
    for job in started["jobs"]:
        result = submitter.poll(job["handle"])
        if result is None:
            return None
        finished.append((job, result))
    statuses = [result["payload"].get("status", "error") for _, result in finished]
    status = "ok" if all(item == "ok" for item in statuses) else next(item for item in statuses if item != "ok")
    payload = {
        "status": status,
        "stderr": "\n".join(
            f"[{job['case_id']}] {result['payload']['stderr']}" for job, result in finished if result["payload"].get("stderr")
        )[-16000:],
        "cases": [case for _, result in finished for case in result["payload"].get("cases") or []],
    }
    submission = {
        "backend": getattr(submitter, "backend", None),
        "jobs": [{"case_id": job["case_id"], **result["submission"]} for job, result in finished],
    }
    return _join(
        payload,
        submission,
        cases=cases,
        references=references,
        baseline_times=baseline_times,
        baseline_energies=baseline_energies,
        factors=factors,
    )
