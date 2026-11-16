"""Submit an evaluation the way a batch system runs one task.

``LocalSubmitter`` is the single-node queue. ``SlurmSubmitter`` writes a batch
script and calls ``sbatch --wait``. Both return only after the job output file
has been written, or the submission itself has failed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobSpec:
    job_id: str
    workdir: Path
    output_path: Path
    timeout_s: float
    env: dict = field(default_factory=dict)


@dataclass
class Submission:
    backend: str
    scheduler_id: str
    returncode: int
    stdout: str
    stderr: str
    script: str


SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def _child_env(spec: JobSpec, root: Path) -> dict[str, str]:
    """Generated programs never see credentials from the harness process."""
    env = {key: value for key, value in os.environ.items() if not _is_secret(key)}
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not previous else str(root) + os.pathsep + previous
    for key, value in spec.env.items():
        if value is None:
            env.pop(key, None)
        elif not _is_secret(key):
            env[key] = str(value)
    return env


def _worker_command(job_file: Path) -> list[str]:
    return [sys.executable, "-m", "esm_harness.eval.worker", "--job", str(job_file)]


class LocalSubmitter:
    backend = "local"

    def __init__(self, root: Path):
        self.root = root

    def submit(self, spec: JobSpec, job_file: Path) -> Submission:
        command = _worker_command(job_file)
        script = " ".join(shlex.quote(part) for part in command)
        try:
            completed = subprocess.run(
                command,
                cwd=spec.workdir,
                env=_child_env(spec, self.root),
                capture_output=True,
                text=True,
                timeout=spec.timeout_s,
                check=False,
            )
            return Submission(
                backend=self.backend,
                scheduler_id=f"local-{spec.job_id}",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                script=script,
            )
        except subprocess.TimeoutExpired as exc:
            return Submission(
                backend=self.backend,
                scheduler_id=f"local-{spec.job_id}",
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\njob exceeded timeout",
                script=script,
            )


class SlurmSubmitter:
    """Submit one evaluation with ``sbatch`` and wait for the task to finish."""

    backend = "slurm"

    def __init__(self, root: Path, partition: str | None = None, account: str | None = None):
        self.root = root
        self.partition = partition
        self.account = account

    def submit(self, spec: JobSpec, job_file: Path) -> Submission:
        command = _worker_command(job_file)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=esm-{spec.job_id}",
            "#SBATCH --cpus-per-task=1",
            f"#SBATCH --time={_slurm_time(spec.timeout_s)}",
            f"#SBATCH --output={spec.workdir / 'slurm.out'}",
            f"#SBATCH --error={spec.workdir / 'slurm.err'}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        memory_mb = spec.env.get("ESM_MEMORY_MB")
        if memory_mb:
            lines.append(f"#SBATCH --mem={memory_mb}")
        lines.extend(
            [
                "set -euo pipefail",
                "for name in $(compgen -e); do",
                '  case "$name" in *API_KEY*|*TOKEN*|*SECRET*|*PASSWORD*) unset "$name" ;; esac',
                "done",
                f"export PYTHONPATH={shlex.quote(str(self.root))}${{PYTHONPATH:+:$PYTHONPATH}}",
                f"cd {shlex.quote(str(spec.workdir))}",
                " ".join(shlex.quote(part) for part in command),
            ]
        )
        script_path = spec.workdir / "submit.sbatch"
        script = "\n".join(lines) + "\n"
        script_path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                ["sbatch", "--wait", str(script_path)],
                cwd=spec.workdir,
                env=_child_env(spec, self.root),
                capture_output=True,
                text=True,
                timeout=spec.timeout_s + 60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return Submission(
                backend=self.backend,
                scheduler_id="",
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nsbatch --wait exceeded timeout",
                script=script,
            )
        scheduler_id = ""
        for token in (completed.stdout or "").split():
            if token.isdigit():
                scheduler_id = token
        return Submission(
            backend=self.backend,
            scheduler_id=scheduler_id,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            script=script,
        )


def make_submitter(config: dict, root: Path):
    backend = config.get("submit", "local")
    if backend == "local":
        return LocalSubmitter(root)
    if backend == "slurm":
        return SlurmSubmitter(
            root,
            partition=config.get("partition"),
            account=config.get("account"),
        )
    raise ValueError(f"unknown submit backend {backend!r}")


def _slurm_time(timeout_s: float) -> str:
    seconds = max(1, int(timeout_s))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
