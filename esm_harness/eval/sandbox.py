"""Prepare a generated program and evaluate it through the job submitter."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from esm_harness.eval.policy import blocked_modules, rejection_reason
from esm_harness.eval.score import summarize_cases
from esm_harness.eval.submit import JobSpec


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
) -> dict:
    reason = rejection_reason(code, list(factors["allowed_imports"]))
    if reason:
        summary = summarize_cases([], factors["speed_weight"], factors["fitness_err_ceiling"])
        return {
            "status": "rejected",
            "stderr": reason,
            "cases": [],
            "submission": None,
            **summary,
        }

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
        "memory_mb": memory_mb,
        "block_imports": blocked_modules(list(factors["allowed_imports"])),
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
    submission = submitter.submit(spec, job_file)
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
    elif submission.returncode == 124:
        payload = {"status": "timeout", "stderr": submission.stderr[-16000:], "cases": []}
    else:
        payload = {
            "status": "error",
            "stderr": (submission.stderr or submission.stdout or "job wrote no result")[-16000:],
            "cases": [],
        }

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
        "submission": {
            "backend": submission.backend,
            "scheduler_id": submission.scheduler_id,
            "returncode": submission.returncode,
            "script": submission.script,
        },
        **summary,
    }
