"""Subprocess runner and acceptance logic for the tutorial harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping

from .models import AcceptanceGate, CandidateRun, MethodSpec, MetricReport


def run_candidate(
    spec: MethodSpec,
    workdir: str | os.PathLike[str] = ".",
    report_dir: str | os.PathLike[str] | None = None,
) -> CandidateRun:
    """Run a candidate with a timeout and a private report path.

    ``workdir`` is the candidate's current directory. ``report_dir`` can keep
    reports in a separate artifact directory without changing relative imports.
    The candidate receives ``HARNESS_REPORT_PATH`` and should write one JSON
    object there. stdout/stderr are retained as evidence for failure triage.
    """

    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_root = (Path(report_dir).resolve() if report_dir is not None else root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{spec.name}.report.json"
    if report_path.exists():
        report_path.unlink()
    env = os.environ.copy()
    env.update(spec.env)
    env["HARNESS_REPORT_PATH"] = str(report_path)
    started = time.perf_counter()
    try:
        command = _resolve_command(spec.command)
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=spec.timeout_s,
            check=False,
        )
        duration = time.perf_counter() - started
        failure_kind = None if completed.returncode == 0 else classify_failure(
            completed.stdout, completed.stderr
        )
        return CandidateRun(
            status="completed",
            command=command,
            returncode=completed.returncode,
            duration_s=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
            report_path=str(report_path) if report_path.exists() else None,
            failure_kind=failure_kind,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        stdout = _decode(exc.stdout)
        stderr = _decode(exc.stderr)
        return CandidateRun(
            status="timeout",
            command=command,
            returncode=None,
            duration_s=duration,
            stdout=stdout,
            stderr=stderr,
            failure_kind="timeout",
        )
    except OSError as exc:
        return CandidateRun(
            status="launch_error",
            command=command,
            returncode=None,
            duration_s=time.perf_counter() - started,
            stdout="",
            stderr=str(exc),
            failure_kind="launch_error",
        )


def load_report(run: CandidateRun) -> MetricReport | None:
    if not run.report_path:
        return None
    try:
        with open(run.report_path, encoding="utf-8") as handle:
            return MetricReport.from_dict(json.load(handle))
    except (OSError, ValueError, TypeError):
        return None


def evaluate_candidate(
    spec: MethodSpec,
    run: CandidateRun,
    report: MetricReport | None = None,
) -> MetricReport:
    """Apply the correctness-first gate described by a task specification."""

    report = report if report is not None else load_report(run)
    performance = spec.performance
    gate = AcceptanceGate(
        correctness_tolerances=spec.correctness_tolerances,
        required_metrics=spec.required_metrics,
        min_speedup=(float(performance["min_speedup"])
                     if "min_speedup" in performance else None),
        max_slowdown=(float(performance["max_slowdown"])
                      if "max_slowdown" in performance else None),
    )
    return gate.evaluate(run, report)


def classify_failure(stdout: str, stderr: str) -> str:
    """Map common candidate failures to stable categories for retry policy."""

    text = f"{stdout}\n{stderr}".lower()
    if "syntaxerror" in text:
        return "syntax_error"
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency_error"
    if "converge" in text or "scf" in text and "error" in text:
        return "scf_failure"
    if "assert" in text or "tolerance" in text or "mismatch" in text:
        return "numerical_failure"
    return "runtime_error"


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _resolve_command(command: tuple[str, ...] | list[str]) -> list[str]:
    """Run Python task commands with the interpreter running the Harness."""

    resolved = list(command)
    if resolved and Path(resolved[0]).name in {"python", "python3"}:
        resolved[0] = sys.executable
    return resolved
