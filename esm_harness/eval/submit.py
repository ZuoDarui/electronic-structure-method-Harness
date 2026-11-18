"""Submit an evaluation the way a batch system runs one task.

``LocalSubmitter`` is the single-node queue. ``SlurmSubmitter`` writes a batch
script and calls ``sbatch --wait``. Both return only after the job output file
has been written, or the submission itself has failed. ``TioneSubmitter`` is
asynchronous: ``start`` creates a TI-ONE task and ``poll`` collects it later.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
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


class SshRemote:
    """Commands and files on the cluster's login node over SSH (paramiko)."""

    def __init__(self, host: str, port: int, user: str, password: str | None = None, key_file: str | None = None):
        self.host, self.port, self.user = host, int(port), user
        self._password, self._key_file = password, key_file
        self._client = None

    def _connect(self):
        if self._client is None:
            import paramiko

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                self.host,
                port=self.port,
                username=self.user,
                password=self._password,
                key_filename=self._key_file,
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
            )
            self._client = client
        return self._client

    def run(self, command: str, timeout: float = 600) -> tuple[int, str, str]:
        _, stdout, stderr = self._connect().exec_command(command, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return stdout.channel.recv_exit_status(), out, err

    def put_text(self, path: str, text: str, mode: int | None = None) -> None:
        sftp = self._connect().open_sftp()
        try:
            with sftp.open(path, "w") as handle:
                handle.write(text)
            if mode is not None:
                sftp.chmod(path, mode)
        finally:
            sftp.close()

    def read_text(self, path: str) -> str | None:
        sftp = self._connect().open_sftp()
        try:
            with sftp.open(path, "r") as handle:
                return handle.read().decode(errors="replace")
        except OSError:
            return None
        finally:
            sftp.close()


_TIONE_STATUS = """\
import json, sys
from pathlib import Path
import tikit
key = json.loads(Path.home().joinpath("secrets/tencentcloud_tione.json").read_text())
client = tikit.Client(key["secret_id"], key["secret_key"], "ap-beijing")
out = {}
for task_id in sys.argv[1:]:
    try:
        detail = client.describe_training_task(task_id).TrainingTaskDetail
        out[task_id] = {"status": detail.Status, "reason": detail.FailureReason or ""}
    except Exception as exc:
        out[task_id] = {"status": "UNKNOWN", "reason": str(exc)[:500]}
print(json.dumps(out))
"""

_TIONE_ENDED = {"FAILED", "STOPPED", "SUCCEED"}

# Runs the user's own submit_training_task.py unchanged, except that the VPC and
# subnet are looked up further back when its own lookup fails: it only reads the
# account's 20 most recent tasks, which on a shared account are often all in
# other resource groups. The network found is cached next to this file.
_TIONE_SUBMIT = """\
import importlib.util, json, sys
from pathlib import Path

CACHE = Path(__file__).with_name("tione_network.json")
SCRIPT = Path(sys.argv.pop(1)).expanduser()
spec = importlib.util.spec_from_file_location("submit_training_task", SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
original = s.resolve_host


def resolve_host(client, host):
    host = dict(host)
    if CACHE.is_file() and s.blank(host.get("vpc")) and s.blank(host.get("subnet")):
        host.update(json.loads(CACHE.read_text()))
    if not s.blank(host.get("vpc")) and not s.blank(host.get("subnet")):
        return original(client, host)
    resolved = original(client, {**host, "vpc": "-", "subnet": "-"})
    for offset in range(0, 2000, 50):
        items = client.describe_training_tasks(offset=offset, limit=50).TrainingTaskSet or []
        for item in items:
            if getattr(item, "ResourceGroupId", None) != resolved["group"]:
                continue
            detail = client.describe_training_task(item.Id).TrainingTaskDetail
            if getattr(detail, "VpcId", None) and getattr(detail, "SubnetId", None):
                network = {"vpc": detail.VpcId, "subnet": detail.SubnetId}
                CACHE.write_text(json.dumps(network))
                return {**resolved, **network}
        if not items:
            break
    s.fail("no training task in this resource group has a VPC and subnet")


s.resolve_host = resolve_host
sys.argv = [str(SCRIPT)] + sys.argv[1:]
s.main()
"""


def _now_s() -> float:
    return time.time()


class TioneSubmitter:
    """One TI-ONE training task per job, created with the cluster's own
    ``submit_training_task.py`` (the script behind ``tione-submit``) through
    ``_TIONE_SUBMIT``, with the resources in the config's ``cluster.task``.

    The job directory goes onto the CFS that every task mounts, the task runs
    ``worker.py`` under ``timeout`` with the cluster's conda environment, and
    the result is read back from CFS. ``start`` returns once the files are
    deployed and a task creation was attempted; ``poll`` retries a failed
    creation and returns None until the task has finished.

    Connection settings come from the run config's ``cluster`` block, and the
    password from ``ESM_CLUSTER_PASSWORD`` (or a key file in
    ``ESM_CLUSTER_KEY_FILE``); ``ESM_CLUSTER_HOST``/``_PORT``/``_USER`` override
    the config.
    """

    backend = "tione"
    asynchronous = True

    def __init__(self, root: Path, cluster: dict, remote=None):
        self.root = root
        self.cluster = dict(cluster)
        self._remote = remote

    @property
    def remote_root(self) -> str:
        return str(self.cluster["remote_root"]).rstrip("/")

    def remote(self):
        if self._remote is None:
            password = os.environ.get("ESM_CLUSTER_PASSWORD")
            key_file = os.environ.get("ESM_CLUSTER_KEY_FILE")
            if not password and not key_file:
                raise RuntimeError("set ESM_CLUSTER_PASSWORD (or ESM_CLUSTER_KEY_FILE) to reach the cluster")
            self._remote = SshRemote(
                os.environ.get("ESM_CLUSTER_HOST") or self.cluster["host"],
                int(os.environ.get("ESM_CLUSTER_PORT") or self.cluster["port"]),
                os.environ.get("ESM_CLUSTER_USER") or self.cluster["user"],
                password=password,
                key_file=key_file,
            )
        return self._remote

    def preflight(self) -> None:
        """Fail before any model call when the cluster cannot take jobs."""
        script = self.cluster["task"]["submit_script"]
        code, out, err = self.remote().run(
            f"bash -lc {shlex.quote(f'mkdir -p {self.remote_root} && test -f {script} && command -v uv')}",
            timeout=120,
        )
        if code != 0:
            raise RuntimeError(f"cluster preflight failed ({code}): {(err or out).strip()[-500:]}")

    def _ensure(self, name: str, text: str) -> str:
        path = f"{self.remote_root}/{name}"
        remote = self.remote()
        if remote.read_text(path) != text:
            remote.put_text(path, text)
        return path

    def _submit(self, handle: dict) -> None:
        """Create the TI-ONE task for a deployed job directory; on failure record why."""
        task = self.cluster["task"]
        wrapper = self._ensure("esm_tione_submit.py", _TIONE_SUBMIT)
        name = f"{self.cluster['user']}-{handle['script'].removesuffix('.sh')}-{time.strftime('%m%d%H%M%S')}"
        args = [
            "uv", "run", "-q", "--with", "tikit", "python", wrapper, str(task["submit_script"]),
            "--host", str(task.get("host", "infra")),
            "--name", name,
            "--gpus", str(task.get("gpus", 1)),
            "--nodes", "1",
            "--cpu", str(task.get("cpu", 48)),
            "--memory", str(task.get("memory", 256)),
            "--image", str(task["image"]),
            "--image-type", str(task.get("image_type", "CCR")),
            "--start-cmd", f"bash {handle['remote_dir']}/{handle['script']}",
            "--submit",
        ]
        handle["submit_attempts"] = int(handle.get("submit_attempts", 0)) + 1
        try:
            code, out, err = self.remote().run(f"bash -lc {shlex.quote(shlex.join(args))}", timeout=600)
        except Exception as exc:
            handle["submit_error"] = f"submit failed: {exc}"[:2000]
            return
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and "task_id" in line:
                handle["task_id"] = json.loads(line)["task_id"]
                handle["task_name"] = name
                handle["submitted"] = _now_s()
                handle.pop("submit_error", None)
                return
        reason = [line for line in (err or out).splitlines() if line.startswith(("submit=", "reason="))]
        handle["submit_error"] = f"submit exited {code}: " + (" ".join(reason) or (err or out).strip()[-2000:])

    def _script(self, timeout_s: float) -> str:
        threads = int(self.cluster.get("threads", 8))
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                'cd "$(dirname "${BASH_SOURCE[0]}")"',
                "for name in $(compgen -e); do",
                '  case "$name" in *API_KEY*|*TOKEN*|*SECRET*|*PASSWORD*) unset "$name" ;; esac',
                "done",
                str(self.cluster["activate"]),
                f"export OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads}",
                'export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1',
                "hostname > host.txt",
                f"timeout --kill-after=60 {int(timeout_s)} python -u worker.py --job job.json > worker.log 2>&1",
                "echo $? > exit_code",
                "",
            ]
        )

    def start(self, spec: JobSpec, job_file: Path, label: str = "") -> dict:
        remote = self.remote()
        remote_dir = f"{self.remote_root}/{spec.job_id}"
        job = json.loads(job_file.read_text(encoding="utf-8"))
        job["program"] = "candidate.py"
        job["output"] = "result.json"
        script_name = f"esm-{spec.job_id}.sh"
        now = _now_s()
        handle = {
            "remote_dir": remote_dir,
            "script": script_name,
            "task_id": None,
            "label": label,
            "first_attempt": now,
            "submitted": now,
            "timeout_s": float(spec.timeout_s),
        }
        remote.run(f"mkdir -p {shlex.quote(remote_dir)}")
        remote.put_text(f"{remote_dir}/candidate.py", (spec.workdir / "candidate.py").read_text(encoding="utf-8"))
        remote.put_text(f"{remote_dir}/job.json", json.dumps(job))
        remote.put_text(f"{remote_dir}/worker.py", (Path(__file__).with_name("worker.py")).read_text(encoding="utf-8"))
        script = self._script(spec.timeout_s)
        remote.put_text(f"{remote_dir}/{script_name}", script, mode=0o755)
        (spec.workdir / script_name).write_text(script, encoding="utf-8")
        self._submit(handle)
        return handle

    def _task_status(self, task_id: str) -> dict:
        helper = f"{self.remote_root}/tione_status.py"
        remote = self.remote()
        if remote.read_text(helper) != _TIONE_STATUS:
            remote.put_text(helper, _TIONE_STATUS)
        try:
            code, out, _ = remote.run(
                f"bash -lc {shlex.quote(f'uv run -q --with tikit python {helper} {task_id}')}", timeout=600
            )
            return json.loads(out.strip().splitlines()[-1])[task_id]
        except Exception as exc:
            return {"status": "UNKNOWN", "reason": f"status query failed: {exc}"[:500]}

    def poll(self, handle: dict) -> dict | None:
        submission = {
            "backend": self.backend,
            "scheduler_id": handle.get("task_id") or "",
            "remote_dir": handle["remote_dir"],
        }
        max_queue_s = float(self.cluster.get("max_queue_s", 86400))
        if not handle.get("task_id"):
            # A failed submission is the cluster's problem, not the program's:
            # retry on each poll, and give up only after max_queue_s.
            self._submit(handle)
            if handle.get("task_id"):
                return None
            if _now_s() - float(handle.get("first_attempt", handle["submitted"])) > max_queue_s:
                return {
                    "payload": {
                        "status": "submit_failed",
                        "stderr": f"no TI-ONE task could be created in {int(max_queue_s)} s "
                        f"({handle['submit_attempts']} attempts): {handle['submit_error']}",
                        "cases": [],
                    },
                    "submission": {**submission, "returncode": None},
                }
            return None
        remote = self.remote()
        exit_text = remote.read_text(f"{handle['remote_dir']}/exit_code")
        if exit_text is not None and exit_text.strip():
            exit_code = int(exit_text.strip())
            result = remote.read_text(f"{handle['remote_dir']}/result.json")
            if result:
                payload = json.loads(result)
            elif exit_code == 124:
                payload = {"status": "timeout", "stderr": f"job exceeded {int(handle['timeout_s'])} s", "cases": []}
            else:
                log = remote.read_text(f"{handle['remote_dir']}/worker.log") or ""
                payload = {"status": "error", "stderr": f"worker exited {exit_code}\n{log[-8000:]}", "cases": []}
            return {"payload": payload, "submission": {**submission, "returncode": exit_code}}
        status = self._task_status(handle["task_id"])
        if status["status"] in _TIONE_ENDED:
            return {
                "payload": {
                    "status": "error",
                    "stderr": f"task {handle['task_id']} ended {status['status']} without a result: {status['reason']}",
                    "cases": [],
                },
                "submission": {**submission, "returncode": None},
            }
        if _now_s() - float(handle["submitted"]) > float(handle["timeout_s"]) + max_queue_s:
            return {
                "payload": {
                    "status": "timeout",
                    "stderr": f"task {handle['task_id']} ({status['status']}) gave no result within "
                    f"{int(handle['timeout_s'] + max_queue_s)} s of submission",
                    "cases": [],
                },
                "submission": {**submission, "returncode": None},
            }
        return None


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
    if backend == "tione":
        return TioneSubmitter(root, config["cluster"])
    raise ValueError(f"unknown submit backend {backend!r}")


def _slurm_time(timeout_s: float) -> str:
    seconds = max(1, int(timeout_s))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
