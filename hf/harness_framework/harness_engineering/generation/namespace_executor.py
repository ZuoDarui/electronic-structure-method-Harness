"""CPU-only SSH executor using Linux unprivileged namespaces and a public image.

This is a container boundary, not a fallback to host Python: candidate code
runs from the exported image after pivot_root, private namespaces, read-only
mounts, capability removal, and no_new_privs. No GPU operations are supported.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
from pathlib import Path
import shlex
import tarfile
import tempfile

from .executor import DockerExecutor, _bounded_process, _PROBE, _REMOTE_PROCESS_IMPLEMENTATION


LAUNCHER = Path(__file__).with_name("namespace_launcher.py")


def _validate_configuration(config):
    if config.get("namespace_protocol") != "cpu-namespace-noproc/1":
        raise ValueError("namespace_protocol must explicitly be cpu-namespace-noproc/1")
    for key in ("namespace_rootfs", "namespace_rootfs_manifest"):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value:
            raise ValueError(f"{key} must be an absolute trusted runtime path")
    sha = config.get("namespace_rootfs_manifest_sha256", "")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("namespace_rootfs_manifest_sha256 must freeze the prepared root filesystem")


class NamespaceExecutor(DockerExecutor):
    def __init__(self, config=None):
        super().__init__(config)
        _validate_configuration(self.config)

    def _launch(self, stage, mode):
        if not self.ssh_host:
            return {"status": "invalid_runtime", "reasons": ["This namespace executor requires an explicit Linux SSH host"]}
        plan = {"mode": mode, "backend": "numpy", "protocol": self.config["namespace_protocol"], "rootfs": self.config["namespace_rootfs"],
                "manifest_path": self.config["namespace_rootfs_manifest"],
                "manifest_sha256": self.config["namespace_rootfs_manifest_sha256"],
                "image_id": self.config.get("cpu_image"), "remote_root": self.config.get("remote_root", "/tmp"),
                "cpus": int(self.cpus), "memory_bytes": self.memory_mb * 1024**2,
                "pids_limit": self.pids_limit, "timeout_s": self.timeout_s,
                "max_log_bytes": self.max_log_bytes, "max_artifact_bytes": self.max_artifact_bytes,
                "probe_code": _PROBE}
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    relative = str(path.relative_to(stage))
                    if relative not in {"candidate/solution.py", "input/problem.npz", "input/options.json", "runner/worker.py"}:
                        raise ValueError("unexpected staged namespace input")
                    archive.add(path, arcname=relative, recursive=False)
            for name, data in (("runner/namespace_launcher.py", LAUNCHER.read_bytes()),
                               ("namespace-plan.json", json.dumps(plan).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        helper = _HELPER.replace("__PROCESS_HELPER__", inspect.getsource(_REMOTE_PROCESS_IMPLEMENTATION))
        process = _bounded_process(self._ssh() + [shlex.join(["python3", "-I", "-c", helper])],
                                   self.timeout_s + 150, self.max_artifact_bytes * 2 + self.max_log_bytes * 4,
                                   input_data=stream.getvalue())
        if process["returncode"] != 0 or process["failure_kind"]:
            return {"status": "invalid_runtime", "reasons": ["CPU namespace transport/bootstrap failed: " + process["stderr"][-2000:]],
                    "transport": {k: v for k, v in process.items() if k != "stdout"}}
        try:
            result = json.loads(process["stdout"])
        except ValueError:
            return {"status": "invalid_runtime", "reasons": ["CPU namespace helper returned an invalid receipt"]}
        if mode == "worker" and result.get("artifacts_base64"):
            total = 0
            for name, encoded in result["artifacts_base64"].items():
                if name not in {"result.npz", "worker.json"}:
                    raise ValueError("unknown namespace output artifact")
                data = base64.b64decode(encoded, validate=True)
                total += len(data)
                if total > self.max_artifact_bytes:
                    raise ValueError("namespace output exceeds aggregate budget")
                (stage / "output" / name).write_bytes(data)
        result.pop("artifacts_base64", None)
        return result

    def preflight(self, backend):
        result = {"status": "invalid_runtime", "backend": backend, "executor": "ssh-usernamespace",
                  "require_external_trace": False, "reference_mounted": False, "reasons": []}
        if backend != "numpy":
            result["reasons"] = ["CPU-only namespace executor; GPU requests are not executed"]
            return result
        try:
            with tempfile.TemporaryDirectory(prefix="hf-namespace-preflight-") as temporary:
                stage = Path(temporary)
                for name in ("candidate", "input", "runner", "output"):
                    (stage / name).mkdir()
                observed = self._launch(stage, "preflight")
            result.update({key: observed[key] for key in ("image", "probe", "isolation", "limits", "rootfs_validation") if key in observed})
            result["status"] = observed["status"]
            result["reasons"] = observed.get("reasons", [])
            result["network"] = "private network namespace with no connected interfaces"
            result["launcher_sha256"] = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as exc:
            result["reasons"] = [type(exc).__name__ + ": " + str(exc)]
        return result

    def _execute_remote(self, stage, backend, runtime):
        if backend != "numpy":
            raise ValueError("CPU-only namespace executor")
        observed = self._launch(stage, "worker")
        if observed.get("status") != "invalid_runtime" and any(observed.get(key) != runtime.get(key) for key in ("image", "probe")):
            observed.update(status="invalid_runtime", reasons=["Actual namespace image/dependency identity differs from its admitted preflight"])
        if observed["status"] == "invalid_runtime":
            return [{"returncode": None, "stdout": "", "stderr": "; ".join(observed.get("reasons", [])),
                     "failure_kind": "isolation_policy", "elapsed_seconds": 0.0, "namespace": observed}]
        process = observed.get("process", {})
        process["namespace"] = {key: observed.get(key) for key in ("image", "probe", "isolation", "limits", "rootfs_validation")}
        return [process]

    def run(self, candidate_dir, problem_path, output_dir, backend="numpy", options=None):
        if backend != "numpy":
            return {"status": "invalid_runtime", "returncode": None, "stdout": "", "stderr": "",
                    "output_path": None, "runtime": self.preflight(backend), "hashes": {}, "trace": {},
                    "reasons": ["CPU-only namespace executor; GPU requests are not executed"]}
        result = super().run(candidate_dir, problem_path, output_dir, backend, options)
        result["executor"] = "ssh-usernamespace"
        if any(process.get("failure_kind") == "isolation_policy" for process in result.get("processes", [])):
            result["status"] = "invalid_runtime"
        if result.get("processes") and (Path(output_dir) / "execution.json").is_file():
            (Path(output_dir) / "execution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result


_HELPER = r'''
import base64,hashlib,io,json,os,platform,signal,stat,subprocess,sys,tarfile,tempfile,time
from pathlib import Path,PurePosixPath
__PROCESS_HELPER__
try:
 data=sys.stdin.buffer.read(512*1024*1024)
 allowed={'candidate/solution.py','input/problem.npz','input/options.json','runner/worker.py','runner/namespace_launcher.py','namespace-plan.json'}
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as archive:
  members=archive.getmembers()
  if len(members)!=len({m.name for m in members}):raise ValueError('duplicate staged namespace paths')
  if any(not m.isfile() or m.name not in allowed for m in members):raise ValueError('unsafe staged namespace input')
  plan=json.load(archive.extractfile('namespace-plan.json'))
  if plan['backend']!='numpy' or platform.machine()!='x86_64' or plan.get('protocol')!='cpu-namespace-noproc/1':raise ValueError('Explicit CPU no-proc namespace executor requires Linux x86_64')
  root=Path(plan['rootfs']);manifest_path=Path(plan['manifest_path'])
  if not root.is_dir() or root.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():raise ValueError('rootfs and manifest must be trusted regular paths')
  manifest_bytes=manifest_path.read_bytes()
  if hashlib.sha256(manifest_bytes).hexdigest()!=plan['manifest_sha256']:raise ValueError('rootfs manifest identity changed')
  manifest=json.loads(manifest_bytes)
  if manifest['schema']!='hf.cpu-rootfs/1' or manifest['source_image_id']!=plan['image_id']:raise ValueError('source image identity differs')
  actual_files=set();actual_links=set()
  for p in root.rglob('*'):
   rel=str(p.relative_to(root))
   if p.is_symlink():
    actual_links.add(rel)
    if manifest['symlinks'].get(rel)!=os.readlink(p):raise ValueError('rootfs link changed: '+rel)
   elif p.is_file():
    actual_files.add(rel)
    with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if manifest['files'].get(rel)!=sha:raise ValueError('rootfs file changed: '+rel)
  if actual_files!=set(manifest['files']) or actual_links!=set(manifest['symlinks']):raise ValueError('rootfs inventory differs')
  with tempfile.TemporaryDirectory(prefix='hf-cpu-namespace-',dir=plan['remote_root']) as directory:
   stage=Path(directory);stage.chmod(0o755)
   for name in ['candidate','input','runner','output']:(stage/name).mkdir(mode=0o755)
   for member in members:
    if member.name=='namespace-plan.json':continue
    path=stage/member.name;path.write_bytes(archive.extractfile(member).read());path.chmod(0o444)
   for name in ['candidate','input']:
    (stage/name/'.readonly_marker').write_text('trusted public read-only marker');(stage/name/'.readonly_marker').chmod(0o666)
   plan.update(stage=str(stage),host_namespaces={name:os.readlink('/proc/self/ns/'+name) for name in ['user','mnt','net','pid','ipc','uts']},affinity=sorted(os.sched_getaffinity(0))[:plan['cpus']])
   if not plan['affinity'] or len(plan['affinity'])!=plan['cpus']:raise ValueError('requested CPU affinity is not available')
   config=stage/'namespace-config.json';config.write_text(json.dumps(plan));config.chmod(0o444)
   command=['unshare','--user','--map-current-user','--keep-caps','--mount','--net','--pid','--ipc','--uts','--fork','--kill-child=SIGKILL','--','python3','-I',str(stage/'runner/namespace_launcher.py'),'--config',str(config)]
   # The trusted bootstrap sees only this explicit environment; no API/SSH key
   # or credential variables are inherited into its namespace or image Python.
   os.environ.clear();os.environ.update({'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','LANG':'C.UTF-8','TMPDIR':'/tmp','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
   process=_bounded_process(command,plan['timeout_s']+90,plan['max_artifact_bytes']*2+plan['max_log_bytes']*2)
   marker='HF_NAMESPACE_RESULT='
   if process['failure_kind'] is not None or process['returncode'] not in [0,1]:raise RuntimeError('namespace bootstrap did not exit cleanly')
   if marker not in process['stdout']:raise RuntimeError('namespace bootstrap did not return a receipt: '+process['stderr'][-2000:])
   result=json.loads(process['stdout'].split(marker,1)[1])
   if (process['returncode']==0)!=(result.get('status')=='passed'):raise RuntimeError('namespace bootstrap exit status and receipt disagree')
   result['image']={'id':manifest['source_image_id'],'rootfs_manifest_sha256':plan['manifest_sha256'],'os':'linux','architecture':'amd64'}
   result['rootfs_validation']={'files_checked':len(actual_files),'symlinks_checked':len(actual_links),'manifest_sha256':plan['manifest_sha256']}
   result['limits']={'memory':'RLIMIT_AS per-process address space; no aggregate cgroup claim','memory_bytes_per_process':plan['memory_bytes'],'cpu':'explicit CPU affinity; no CPU quota claim','affinity':plan['affinity'],'pids':'RLIMIT_NPROC per real UID','pids_limit':plan['pids_limit'],'output':'private tmpfs aggregate byte cap plus RLIMIT_FSIZE','output_bytes':plan['max_artifact_bytes'],'wall_timeout_seconds':plan['timeout_s'],'no_host_python_candidate_execution':True}
 print(json.dumps(result,allow_nan=False,separators=(',',':')))
except Exception as e:
 print(json.dumps({'status':'invalid_runtime','reasons':[type(e).__name__+': '+str(e)]}))
'''
