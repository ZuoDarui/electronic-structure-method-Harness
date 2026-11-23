"""Fail-closed Docker execution for generated RHF programs.

Only candidate source, integral inputs, a trusted worker, and options are sent
to the container. The reference and verifier are never mounted. Docker images
are explicitly prebuilt; execution never pulls an image or falls back to a host
Python process. SSH transports the same isolated job to an existing Docker host.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import uuid


STAGES = ("hf.jk0", "hf.jk1", "hf.jksum", "hf.solve")
_WORKER = Path(__file__).with_name("worker.py")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded_process(command, timeout_s, max_log_bytes, input_data=None, binary=False):
    """Drain to temporary files with a hard capture limit; kill timed-out clients."""
    started = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as inp:
        if input_data is not None:
            inp.write(input_data)
            inp.seek(0)
        try:
            process = subprocess.Popen(command, stdin=inp if input_data is not None else subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
        except OSError as exc:
            return {"returncode": None, "stdout": b"" if binary else "", "stderr": str(exc),
                    "failure_kind": "launch_error", "elapsed_seconds": time.monotonic() - started}
        reason = None
        while process.poll() is None:
            if time.monotonic() - started > timeout_s:
                reason = "timeout"
            elif os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > max_log_bytes:
                reason = "log_limit_exceeded"
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.025)
        process.wait()
        if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > max_log_bytes:
            reason = reason or "log_limit_exceeded"
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(max_log_bytes), err.read(max_log_bytes)
        return {"returncode": process.returncode,
                "stdout": stdout if binary else stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"), "failure_kind": reason,
                "elapsed_seconds": time.monotonic() - started}


_REMOTE_PROCESS_IMPLEMENTATION = _bounded_process


def check_container_policy(raw, run_arguments):
    """Compare actual daemon policy to requested isolation before any code runs.

    Managed Docker gateways may accept flags while silently replacing them.
    CLI success and failed write/network probes therefore do not establish
    isolation. This check must precede start for every probe and candidate.
    """
    host, config = raw.get("HostConfig", {}), raw.get("Config", {})
    expected, requested_mounts = {}, []
    for index, token in enumerate(run_arguments):
        if token in {"--memory", "--memory-swap", "--cpus", "--pids-limit", "--user", "--mount", "--tmpfs"}:
            if index + 1 >= len(run_arguments):
                raise ValueError("invalid trusted Docker argument list")
            value = run_arguments[index + 1]
            if token == "--mount":
                fields = dict(part.split("=", 1) for part in value.split(",") if "=" in part)
                requested_mounts.append((fields["src"], fields["dst"], "readonly" not in value.split(",")))
            elif token == "--tmpfs":
                expected["tmpfs"] = value
            else:
                expected[token] = value
    reasons = []
    checks = {
        "NetworkMode": host.get("NetworkMode") == "none",
        "ReadonlyRootfs": host.get("ReadonlyRootfs") is True,
        "Privileged": host.get("Privileged") is False,
        "CapDrop": "ALL" in {str(value).upper() for value in (host.get("CapDrop") or [])},
        "SecurityOpt": any(value in {"no-new-privileges", "no-new-privileges:true"} for value in (host.get("SecurityOpt") or [])),
        "Init": host.get("Init") is True,
        "PidsLimit": host.get("PidsLimit") == int(expected["--pids-limit"]),
        "Memory": host.get("Memory") == int(expected["--memory"].removesuffix("m")) * 1024**2,
        "MemorySwap": host.get("MemorySwap") == int(expected["--memory-swap"].removesuffix("m")) * 1024**2,
        "NanoCpus": host.get("NanoCpus") == int(float(expected["--cpus"]) * 10**9),
        "User": config.get("User") == expected["--user"],
    }
    tmpfs = host.get("Tmpfs") or {}
    target, options = expected["tmpfs"].split(":", 1)
    checks["Tmpfs"] = target in tmpfs and set(options.split(",")) <= set(tmpfs[target].split(","))
    observed_mounts = [(m.get("Source"), m.get("Destination"), m.get("RW"))
                       for m in raw.get("Mounts", []) if m.get("Type") == "bind"]
    checks["BindMounts"] = sorted(observed_mounts) == sorted(requested_mounts)
    requested_images = [value for value in run_arguments if isinstance(value, str) and value.startswith("sha256:")]
    checks["Image"] = len(requested_images) == 1 and raw.get("Image") == requested_images[0]
    for key, valid in checks.items():
        if not valid:
            reasons.append(f"Docker daemon did not preserve required isolation: {key}")
    observed = {key: host.get(key) for key in ("NetworkMode", "ReadonlyRootfs", "Privileged", "CapDrop", "SecurityOpt",
                                               "Init", "PidsLimit", "Memory", "MemorySwap", "NanoCpus", "Tmpfs")}
    observed.update(User=config.get("User"), Image=raw.get("Image"), bind_mounts=observed_mounts)
    return {"valid": not reasons, "reasons": reasons, "observed": observed}


def _checked_docker_process(command, docker, timeout_s, max_log_bytes, invoke=None):
    """Create, inspect, then start; a rewritten policy never starts the payload."""
    if invoke is None:
        invoke = lambda args, timeout=30: _bounded_process(docker + args, timeout, max_log_bytes)
    arguments = command[len(docker):]
    if arguments[0] != "run" or "--name" not in arguments:
        raise ValueError("checked Docker execution requires a named run command")
    name = arguments[arguments.index("--name") + 1]
    create_arguments = ["create"] + [argument for argument in arguments[1:] if argument != "--rm"]
    record = {"returncode": None, "stdout": "", "stderr": "", "failure_kind": "isolation_policy", "elapsed_seconds": 0.0}
    try:
        created = invoke(create_arguments, timeout=30)
        if created["returncode"] != 0 or created["failure_kind"]:
            return {**created, "failure_kind": "isolation_policy"}
        inspected = invoke(["inspect", name], timeout=20)
        if inspected["returncode"] != 0 or inspected["failure_kind"]:
            return {**inspected, "failure_kind": "isolation_policy"}
        raw = json.loads(inspected["stdout"])[0]
        policy = check_container_policy(raw, arguments)
        record["isolation_policy"] = policy
        if not policy["valid"]:
            record["stderr"] = "; ".join(policy["reasons"])
            return record
        record = invoke(["start", "--attach", name], timeout=timeout_s)
        record["isolation_policy"] = policy
        if not record["failure_kind"]:
            state = invoke(["inspect", name], timeout=20)
            if state["returncode"] != 0 or state["failure_kind"]:
                record.update(returncode=None, failure_kind="runtime_receipt", stderr=record["stderr"] + "\nCannot obtain final container state")
            else:
                final = json.loads(state["stdout"])[0]["State"]
                if final.get("Running") is not False or type(final.get("ExitCode")) is not int:
                    record.update(returncode=None, failure_kind="runtime_receipt")
                else:
                    record["returncode"] = final["ExitCode"]
        return record
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        record.update(returncode=None, failure_kind="isolation_policy", stderr="Invalid Docker isolation receipt: " + str(exc))
        return record
    finally:
        invoke(["rm", "-f", name], timeout=10)


def validate_external_trace(path, timeout_s=15):
    """Require real CUDA kernels linked to launches inside every trusted NVTX stage.

    A total kernel count is insufficient: it could only describe import/probe
    activity. CUPTI correlationId and globalTid bind launches to the stage; kernel
    execution must also fit inside its synchronized NVTX interval. This proves
    observed device activity, not that a malicious candidate computed honestly.
    """
    verdict = {"valid": False, "source": "nsys CUDA/CUPTI SQLite + NVTX",
               "scope": "per-stage CUDA kernel execution; not an adversarial proof of algorithm intent",
               "stage_coverage": {}, "reasons": []}
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        verdict["reasons"].append("external CUDA trace is missing or is a symlink")
        return verdict
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        deadline = time.monotonic() + timeout_s
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kernel_table = next((t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in tables), None)
        if "NVTX_EVENTS" not in tables or kernel_table is None:
            raise ValueError("trace lacks NVTX or CUDA kernel records")
        def columns(table):
            return {r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')}
        nvtx_columns, kernel_columns = columns("NVTX_EVENTS"), columns(kernel_table)
        if not {"start", "end", "globalTid"} <= nvtx_columns:
            raise ValueError("NVTX schema lacks timestamps or thread identity")
        if not {"start", "end", "correlationId"} <= kernel_columns:
            raise ValueError("CUDA kernel schema lacks timestamps or launch correlation")
        strings = {}
        if "StringIds" in tables:
            strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        text_col = '"text"' if "text" in nvtx_columns else "NULL"
        id_col = '"textId"' if "textId" in nvtx_columns else "NULL"
        ranges = []
        for start, end, tid, text, text_id in connection.execute(
                f'SELECT start, end, globalTid, {text_col}, {id_col} FROM NVTX_EVENTS WHERE end IS NOT NULL'):
            label = text or strings.get(text_id)
            if label in STAGES and isinstance(start, int) and isinstance(end, int) and end >= start:
                ranges.append((label, start, end, tid))
        for stage in STAGES:
            verdict["stage_coverage"][stage] = {"range_count": sum(r[0] == stage for r in ranges),
                                                  "kernel_count": 0, "kernel_nanoseconds": 0,
                                                  "kernel_names": []}
        if len(ranges) > 10000:
            raise ValueError("unreasonable number of candidate-stage ranges")
        launch_tables = [t for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
                         if t in tables and {"start", "globalTid", "correlationId"} <= columns(t)]
        if not launch_tables:
            raise ValueError("trace lacks correlated CUDA launch/thread records")
        name_col = next((c for c in ("demangledName", "shortName", "name") if c in kernel_columns), None)
        name_expr = f'k."{name_col}"' if name_col else "NULL"
        seen = set()
        for launches in launch_tables:
            for label, start, end, tid in ranges:
                process_condition = ' AND (k.globalPid >> 24)=(r.globalTid >> 24)' if "globalPid" in kernel_columns else ''
                query = (f'SELECT k.start, k.end, k.correlationId, {name_expr} FROM "{kernel_table}" k '
                         f'JOIN "{launches}" r ON k.correlationId=r.correlationId '
                         'WHERE r.globalTid=? AND r.start>=? AND r.start<=? AND k.start>=? AND k.end<=?' + process_condition)
                for ks, ke, correlation, name in connection.execute(query, (tid, start, end, start, end)):
                    identity = (label, ks, ke, correlation)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    item = verdict["stage_coverage"][label]
                    item["kernel_count"] += 1
                    item["kernel_nanoseconds"] += max(0, ke - ks)
                    decoded = strings.get(name, name)
                    if decoded is not None and len(item["kernel_names"]) < 8 and str(decoded) not in item["kernel_names"]:
                        item["kernel_names"].append(str(decoded))
        for stage, coverage in verdict["stage_coverage"].items():
            if coverage["range_count"] == 0 or coverage["kernel_count"] == 0:
                verdict["reasons"].append(f"no correlated CUDA kernel inside {stage}")
        verdict["valid"] = not verdict["reasons"]
        verdict["sha256"] = _hash(path)
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        verdict["reasons"].append(f"invalid external CUDA trace: {exc}")
    finally:
        if connection is not None:
            connection.close()
    return verdict


_PROBE = r'''
import hashlib,importlib,importlib.util,json,os,platform,sys
from pathlib import Path
import numpy, scipy
if os.geteuid()==0: raise RuntimeError('container must run as a non-root user')
if importlib.util.find_spec('pyscf') is not None: raise RuntimeError('candidate image must not include PySCF/reference software')
def loaded_identity(name):
 module=importlib.import_module(name)
 path=Path(module.__file__).resolve()
 if not path.is_file(): raise RuntimeError('loaded module has no regular file identity: '+name)
 with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'sha256':digest}
modules={name:loaded_identity(name) for name in ('numpy','scipy','scipy.linalg._flapack')}
for extension_name in ('numpy._core._multiarray_umath','numpy.core._multiarray_umath'):
 try:
  extension=loaded_identity(extension_name)
  if extension['path'].endswith(('.so','.pyd')):modules['numpy._core._multiarray_umath']=extension;break
 except ImportError:pass
else:raise RuntimeError('cannot identify the loaded NumPy core extension')
cpu_values,_=numpy.linalg.eigh(numpy.asarray([[2.,0.],[0.,3.]],dtype=numpy.float64))
if cpu_values.tolist()!=[2.,3.]:raise RuntimeError('NumPy float64 eigensolver failed')
result={'python':sys.version,'platform':platform.platform(),'uid':os.geteuid(),'numpy':numpy.__version__,'scipy':scipy.__version__,'pyscf_available':False,'loaded_modules':modules,'numpy_float64_eigh':cpu_values.tolist()}
if sys.argv[1]=='cupy':
 import cupy as cp
 if cp.cuda.runtime.getDeviceCount()!=1: raise RuntimeError('exactly one selected CUDA device must be visible')
 a=cp.asarray([[2.,0.],[0.,3.]],dtype=cp.float64);v,_=cp.linalg.eigh(a)
 cp.cuda.get_current_stream().synchronize()
 if cp.asnumpy(v).tolist()!=[2.,3.]: raise RuntimeError('CUDA float64 eigensolver failed')
 name=cp.cuda.runtime.getDeviceProperties(0)['name']
 result.update(cupy=cp.__version__,cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()),cuda_driver=int(cp.cuda.runtime.driverGetVersion()),device_name=name.decode() if isinstance(name,bytes) else str(name),device_count=1)
 for module_name in ('cupy','cupy._core.core'):result['loaded_modules'][module_name]=loaded_identity(module_name)
print(json.dumps(result))
'''


class DockerExecutor:
    def __init__(self, config=None):
        self.config = dict(config or {})
        command = self.config.get("docker_command", ["docker"])
        self.docker = [command] if isinstance(command, str) else list(command)
        if not self.docker or any(not isinstance(v, str) or not v for v in self.docker):
            raise ValueError("docker_command must be a nonempty argument list")
        self.timeout_s = self._positive("timeout_s", 180)
        self.max_log_bytes = int(self._positive("max_log_bytes", 1024 * 1024))
        self.max_artifact_bytes = int(self._positive("max_artifact_bytes", 256 * 1024 * 1024))
        if "memory_mb" not in self.config and "memory_limit_mb" in self.config:
            self.config["memory_mb"] = self.config["memory_limit_mb"]
        self.memory_mb = int(self._positive("memory_mb", 2048))
        self.cpus = self._positive("cpus", 1)
        self.pids_limit = int(self._positive("pids_limit", 64))
        for key in ("max_log_bytes", "max_artifact_bytes", "memory_mb", "pids_limit"):
            value = getattr(self, key)
            if value < 1 or (key in self.config and self.config[key] != value):
                raise ValueError(f"{key} must be a positive integer")
        if "require_external_trace" in self.config and type(self.config["require_external_trace"]) is not bool:
            raise ValueError("require_external_trace must be boolean")
        self.user = str(self.config.get("user", "65532:65532"))
        if not re.fullmatch(r"[1-9][0-9]*(?::[1-9][0-9]*)?", self.user):
            raise ValueError("container user must be an explicitly non-root numeric uid[:gid]")
        self.ssh_host = self.config.get("ssh_host")
        if self.ssh_host and (not isinstance(self.ssh_host, str) or
                              not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", self.ssh_host)):
            raise ValueError("invalid ssh_host")
        self.ssh_port = int(self.config.get("ssh_port", 22))
        if not 1 <= self.ssh_port <= 65535:
            raise ValueError("invalid ssh_port")
        self.nsys = str(self.config.get("nsys_path", "nsys"))
        if not self.nsys or self.nsys.startswith("-"):
            raise ValueError("invalid nsys_path")

    def _positive(self, key, default):
        value = self.config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return float(value)

    def _ssh(self):
        options = self.config.get("ssh_options", [])
        if not isinstance(options, list) or any(not isinstance(v, str) for v in options):
            raise ValueError("ssh_options must be an argument list")
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=10", "-p", str(self.ssh_port), *options, "--", self.ssh_host]

    def _docker(self, args, timeout=30):
        command = self.docker + list(args)
        if self.ssh_host:
            command = self._ssh() + [shlex.join(command)]
        return _bounded_process(command, timeout, self.max_log_bytes)

    def _flags(self, backend, name):
        flags = ["run", "--rm", "--pull=never", "--name", name, "--init", "--network=none",
                 "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                 "--pids-limit", str(self.pids_limit), "--memory", f"{self.memory_mb}m",
                 "--memory-swap", f"{self.memory_mb}m", "--cpus", str(self.cpus),
                 "--user", self.user, "--ulimit", "core=0", "--ulimit", "nofile=256:256",
                 "--ulimit", f"fsize={self.max_artifact_bytes}:{self.max_artifact_bytes}",
                 "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777", "--workdir", "/tmp"]
        for key, value in {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "CUPY_CACHE_DIR": "/tmp/cupy-cache", "CUDA_CACHE_PATH": "/tmp/cuda-cache"}.items():
            flags += ["--env", f"{key}={value}"]
        if backend == "cupy":
            device = str(self.config.get("gpu_device", "0"))
            if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", device):
                raise ValueError("gpu_device must select one physical device")
            flags += ["--gpus", f"device={device}"]
            flags += ["--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
            mount = self.config.get("nsys_mount")
            if mount:
                source, target = str(mount["source"]), str(mount.get("target", "/opt/nsys"))
                if not Path(source).is_absolute() or not target.startswith("/opt/") or any(c in source + target for c in (",", "\n", "\x00")):
                    raise ValueError("nsys_mount must map an absolute trusted directory into /opt")
                flags += ["--mount", f"type=bind,src={source},dst={target},readonly"]
        else:
            flags += ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
        return flags

    def preflight(self, backend):
        result = {"status": "invalid_runtime", "backend": backend,
                  "executor": "ssh-docker" if self.ssh_host else "docker",
                  "network": "none", "reference_mounted": False, "reasons": []}
        if backend not in {"numpy", "cupy"}:
            result["reasons"].append("backend must be numpy or cupy; fallback is forbidden")
            return result
        external = bool(self.config.get("require_external_trace", True)) and backend == "cupy"
        result["require_external_trace"] = external
        image = self.config.get("gpu_image" if backend == "cupy" else "cpu_image",
                                "hf-harness-gpu:local" if backend == "cupy" else "hf-harness-cpu:local")
        if not isinstance(image, str) or not image or image.startswith("-"):
            result["reasons"].append("invalid configured image")
            return result
        result["configured_image"] = image
        try:
            version = self._docker(["version", "--format", "{{json .Server}}"])
            if version["returncode"] != 0 or version["failure_kind"]:
                raise RuntimeError("Docker CLI/daemon unavailable: " + version["stderr"][:500])
            server = json.loads(version["stdout"])
            if str(server.get("Os", server.get("OS", ""))).lower() != "linux":
                raise RuntimeError("Linux Docker containers are required")
            result["docker_server"] = {k: server.get(k) for k in ("Version", "Os", "Arch", "ApiVersion")}
            inspected = self._docker(["image", "inspect", image, "--format",
                                      '{"id":{{json .Id}},"os":{{json .Os}},"architecture":{{json .Architecture}},"repo_digests":{{json .RepoDigests}}}'])
            if inspected["returncode"] != 0 or inspected["failure_kind"]:
                raise RuntimeError("prebuilt candidate image is unavailable; automatic image pulls are disabled")
            identity = json.loads(inspected["stdout"])
            if identity.get("os") != "linux" or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity.get("id", "")):
                raise RuntimeError("image lacks a Linux content identity")
            result["image"] = identity
            name = "hf-preflight-" + uuid.uuid4().hex
            probe = _checked_docker_process(self.docker + self._flags(backend, name) + [identity["id"], "python3", "-I", "-c", _PROBE, backend], self.docker, 90, self.max_log_bytes, self._docker)
            if probe["returncode"] != 0 or probe["failure_kind"]:
                result["isolation_policy"] = probe.get("isolation_policy")
                raise RuntimeError("isolated runtime probe failed: " + probe["stderr"][:1500])
            result["probe"] = json.loads(probe["stdout"].strip().splitlines()[-1])
            result["isolation_policy"] = probe["isolation_policy"]
            if external:
                name = "hf-nsys-preflight-" + uuid.uuid4().hex
                nsys = _checked_docker_process(self.docker + self._flags(backend, name) + [identity["id"], self.nsys, "--version"], self.docker, 30, self.max_log_bytes, self._docker)
                if nsys["returncode"] != 0 or nsys["failure_kind"]:
                    raise RuntimeError("strict GPU verification requires nsys inside the trusted image or a trusted read-only nsys_mount")
                result["nsys_version"] = (nsys["stdout"] + nsys["stderr"]).strip()
            result["status"] = "passed"
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            result["reasons"].append(str(exc))
        return result

    def _commands(self, stage, backend, runtime):
        name = "hf-candidate-" + uuid.uuid4().hex
        flags = self._flags(backend, name)
        for source, target in (("candidate", "/candidate"), ("input", "/input"), ("runner", "/runner")):
            path = str(Path(stage) / source)
            if "," in path or "\n" in path:
                raise ValueError("mount path contains unsupported characters")
            flags += ["--mount", f"type=bind,src={path},dst={target},readonly"]
        flags += ["--mount", f"type=bind,src={Path(stage) / 'output'},dst=/output"]
        worker = ["python3", "-I", "/runner/worker.py", "--candidate", "/candidate/solution.py",
                  "--problem", "/input/problem.npz", "--output", "/output/result.npz",
                  "--backend", backend, "--options", "/input/options.json"]
        if runtime["require_external_trace"]:
            worker = [self.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                      "--force-overwrite=true", "--output=/output/cuda_trace", *worker]
        commands = [self.docker + flags + [runtime["image"]["id"], *worker]]
        names = [name]
        if runtime["require_external_trace"]:
            export_name = "hf-export-" + uuid.uuid4().hex
            names.append(export_name)
            export_flags = list(flags)
            export_flags[export_flags.index(name)] = export_name
            commands.append(self.docker + export_flags + [runtime["image"]["id"], self.nsys, "export",
                            "--type=sqlite", "--force-overwrite=true", "--output=/output/cuda_trace.sqlite",
                            "/output/cuda_trace.nsys-rep"])
        return commands, names

    def _execute_local(self, stage, commands, names):
        records = []
        try:
            for command in commands:
                record = _checked_docker_process(command, self.docker, self.timeout_s, self.max_log_bytes)
                records.append(record)
                if record["returncode"] != 0 or record["failure_kind"]:
                    break
        finally:
            for name in names:
                self._docker(["rm", "-f", name], timeout=10)
        return records

    def _execute_remote(self, stage, backend, runtime):
        """Ship no references; the remote helper is trusted and never runs candidates outside Docker."""
        remote_root = str(self.config.get("remote_root", "/tmp"))
        if not remote_root.startswith("/") or "\x00" in remote_root:
            raise ValueError("remote_root must be an absolute existing directory")
        commands, names = self._commands("__HF_STAGE__", backend, runtime)
        plan = {"commands": commands, "names": names, "docker": self.docker,
                "timeout_s": self.timeout_s, "max_log_bytes": self.max_log_bytes,
                "max_artifact_bytes": self.max_artifact_bytes, "remote_root": remote_root}
        (stage / "execution-plan.json").write_text(json.dumps(plan))
        bundle = io.BytesIO()
        with tarfile.open(fileobj=bundle, mode="w") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(stage)), recursive=False)
        # Embed the identical bounded process helper rather than depending on a
        # separately installed remote harness package.
        import inspect
        helper = _REMOTE_HELPER.replace("__BOUNDED_PROCESS__", inspect.getsource(_REMOTE_PROCESS_IMPLEMENTATION))
        helper = helper.replace("__POLICY_CHECK__", inspect.getsource(check_container_policy))
        helper = helper.replace("__CHECKED_DOCKER__", inspect.getsource(_checked_docker_process))
        helper = helper.replace("__REMOTE_ROOT__", repr(remote_root))
        command = self._ssh() + [shlex.join(["python3", "-c", helper])]
        result = _bounded_process(command, self.timeout_s * len(commands) + 90,
                                  self.max_artifact_bytes + self.max_log_bytes * 4,
                                  input_data=bundle.getvalue(), binary=True)
        if result["returncode"] != 0 or result["failure_kind"]:
            result["stdout"] = ""
            return [result]
        total = 0
        process_records = None
        with tarfile.open(fileobj=io.BytesIO(result["stdout"]), mode="r:") as archive:
            for member in archive:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe artifact in SSH result archive")
                total += member.size
                if total > self.max_artifact_bytes:
                    raise ValueError("SSH artifacts exceed size limit")
                data = archive.extractfile(member).read()
                if member.name == "process.json":
                    process_records = json.loads(data)
                elif len(relative.parts) == 2 and relative.parts[0] == "output":
                    (stage / relative).write_bytes(data)
                else:
                    raise ValueError("unexpected SSH result artifact")
        if not isinstance(process_records, list):
            raise ValueError("SSH result lacks process execution receipt")
        return process_records

    def run(self, candidate_dir, problem_path, output_dir, backend="numpy", options=None):
        candidate_dir, problem_path, output_dir = Path(candidate_dir), Path(problem_path), Path(output_dir)
        result = {"status": "invalid_runtime", "returncode": None, "stdout": "", "stderr": "",
                  "output_path": None, "runtime": {}, "hashes": {}, "trace": {}, "reasons": []}
        candidate = candidate_dir / "solution.py"
        own_output = False
        try:
            if candidate.is_symlink() or not candidate.is_file() or problem_path.is_symlink() or not problem_path.is_file():
                raise ValueError("candidate and problem must be regular, non-symlink files")
            if candidate.stat().st_size > 1024 * 1024 or problem_path.stat().st_size > self.max_artifact_bytes:
                raise ValueError("candidate/input exceeds configured size limit")
            if output_dir.is_symlink() or (output_dir.exists() and any(output_dir.iterdir())):
                raise ValueError("output directory must be fresh and empty; stale results are forbidden")
            options_text = json.dumps(dict(options or {}), allow_nan=False, sort_keys=True)
            result["hashes"] = {"candidate_sha256": _hash(candidate), "input_sha256": _hash(problem_path),
                                 "worker_sha256": _hash(_WORKER), "options_sha256": hashlib.sha256(options_text.encode()).hexdigest()}
            result["input_sha256"] = result["hashes"]["input_sha256"]
            result["candidate_sha256"] = result["hashes"]["candidate_sha256"]
            runtime = self.preflight(backend)
            result["runtime"] = runtime
            if runtime["status"] != "passed":
                result["reasons"] = runtime["reasons"]
                return result
            output_dir.mkdir(parents=True, exist_ok=True)
            own_output = True
            with tempfile.TemporaryDirectory(prefix="hf-isolated-") as temporary:
                stage = Path(temporary)
                for name in ("candidate", "input", "runner", "output"):
                    (stage / name).mkdir(mode=0o755)
                (stage / "output").chmod(0o777)
                for source, dest in ((candidate, "candidate/solution.py"), (problem_path, "input/problem.npz"), (_WORKER, "runner/worker.py")):
                    shutil.copyfile(source, stage / dest)
                    (stage / dest).chmod(0o444)
                (stage / "input/options.json").write_text(options_text)
                (stage / "input/options.json").chmod(0o444)
                if self.ssh_host:
                    processes = self._execute_remote(stage, backend, runtime)
                else:
                    commands, names = self._commands(stage, backend, runtime)
                    processes = self._execute_local(stage, commands, names)
                result["processes"] = processes
                result["returncode"] = processes[-1]["returncode"] if processes else None
                result["stdout"] = "\n".join(p["stdout"] for p in processes)[:self.max_log_bytes]
                result["stderr"] = "\n".join(p["stderr"] for p in processes)[:self.max_log_bytes]
                failed = not processes or any(p["returncode"] != 0 or p["failure_kind"] for p in processes)
                result["status"] = "failed" if failed else "passed"
                if failed:
                    result["reasons"].append("isolated candidate or trace collector failed")
                    if any(p.get("failure_kind") == "isolation_policy" for p in processes):
                        result["status"] = "invalid_runtime"
                total = 0
                allowed = {"result.npz", "worker.json", "cuda_trace.nsys-rep", "cuda_trace.sqlite"}
                for path in (stage / "output").iterdir():
                    if path.is_symlink() or not path.is_file() or path.name not in allowed:
                        raise ValueError("container produced an unexpected output artifact")
                    total += path.stat().st_size
                    if total > self.max_artifact_bytes:
                        raise ValueError("container artifacts exceed size limit")
                    shutil.copyfile(path, output_dir / path.name)
                worker_path = output_dir / "worker.json"
                worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
                result["worker"] = worker
                for key in ("candidate_sha256", "input_sha256"):
                    if worker.get(key) != result["hashes"][key] or worker.get(key + "_after", worker.get(key)) != result["hashes"][key]:
                        result["status"] = "failed"
                        result["reasons"].append(f"worker {key} differs from frozen input")
                result_file = output_dir / "result.npz"
                if not result_file.is_file() or worker.get("status") != "passed":
                    result["status"] = "failed"
                    result["reasons"].append("worker did not produce a complete result")
                elif worker.get("output_sha256") != _hash(result_file):
                    result["status"] = "failed"
                    result["reasons"].append("result differs from worker output receipt")
                else:
                    result["output_path"] = str(result_file.resolve())
                if backend == "cupy":
                    result["trace"]["cooperative"] = worker.get("gpu_trace", {})
                    if runtime["require_external_trace"]:
                        trace = validate_external_trace(output_dir / "cuda_trace.sqlite")
                        result["trace"]["external"] = trace
                        if not trace["valid"]:
                            if result["status"] == "passed":
                                result["status"] = "invalid_runtime"
                            result["reasons"].extend(trace["reasons"])
                        solve = worker.get("gpu_trace", {}).get("stage_coverage", {}).get("hf.solve", {})
                        if not solve.get("eigensolver_count", 0):
                            result["status"] = "failed"
                            result["reasons"].append("no device eigensolver was observed inside solve; returning a GPU array alone is insufficient")
                    elif not worker.get("gpu_trace", {}).get("gpu_executed"):
                        result["status"] = "failed"
                        result["reasons"].append("cooperative GPU stage coverage is incomplete")
                result["hashes"]["artifacts"] = {p.name: _hash(p) for p in output_dir.iterdir() if p.is_file()}
                if any(p.get("failure_kind") == "isolation_policy" for p in processes):
                    result["status"] = "invalid_runtime"
            result["isolated_stage_removed"] = True
            # The original files are also checked; the isolated copies were RO.
            if _hash(candidate) != result["hashes"]["candidate_sha256"] or _hash(problem_path) != result["hashes"]["input_sha256"]:
                result["status"] = "failed"
                result["reasons"].append("controller inputs changed during execution")
        except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as exc:
            result["status"] = "failed" if result["runtime"].get("status") == "passed" else "invalid_runtime"
            result["reasons"].append(f"{type(exc).__name__}: {exc}")
        if own_output and output_dir.is_dir() and not output_dir.is_symlink():
            (output_dir / "execution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result


# Only this trusted helper runs on the SSH host. Candidate bytes are unpacked
# into a new private directory and are executed exclusively by the Docker job.
_REMOTE_HELPER = r'''
import io,json,os,signal,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
__BOUNDED_PROCESS__
__POLICY_CHECK__
__CHECKED_DOCKER__
bundle=sys.stdin.buffer.read(512*1024*1024)
allowed={'candidate/solution.py','input/problem.npz','input/options.json','runner/worker.py','execution-plan.json'}
payload=None
with tempfile.TemporaryDirectory(prefix='hf-isolated-',dir=__REMOTE_ROOT__) as temporary:
 root=Path(temporary)
 with tarfile.open(fileobj=io.BytesIO(bundle),mode='r:') as archive:
  for member in archive:
   if not member.isfile() or member.name not in allowed: raise ValueError('unsafe staged file')
   path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read());path.chmod(0o444)
 plan=json.loads((root/'execution-plan.json').read_text())
 (root/'output').mkdir(mode=0o777);(root/'output').chmod(0o777)
 root.chmod(0o755)
 records=[]
 try:
  for command in plan['commands']:
   command=[part.replace('__HF_STAGE__',str(root)) for part in command]
   record=_checked_docker_process(command,plan['docker'],plan['timeout_s'],plan['max_log_bytes']);records.append(record)
   if record['returncode']!=0 or record['failure_kind']:break
 finally:
  for name in plan['names']:_bounded_process(plan['docker']+['rm','-f',name],10,65536)
 result=io.BytesIO()
 with tarfile.open(fileobj=result,mode='w') as archive:
  encoded=json.dumps(records).encode();info=tarfile.TarInfo('process.json');info.size=len(encoded);archive.addfile(info,io.BytesIO(encoded))
  total=0
  for path in (root/'output').iterdir():
   if path.is_symlink() or not path.is_file() or path.name not in {'result.npz','worker.json','cuda_trace.nsys-rep','cuda_trace.sqlite'}:raise ValueError('unexpected candidate output')
   total+=path.stat().st_size
   if total>plan['max_artifact_bytes']:raise ValueError('artifact size limit exceeded')
   archive.add(path,arcname='output/'+path.name,recursive=False)
 payload=result.getvalue()
sys.stdout.buffer.write(payload)
'''
