"""Trusted CPU user-namespace bootstrap; never import a candidate here.

The setup process starts with capabilities in its new user namespace only.
It pivots into a selected public root filesystem, drops every capability and
sets no_new_privs before starting the image's Python. Outputs live on a bounded
private tmpfs and are collected only after all candidate descendants are dead.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time


_LIBC = ctypes.CDLL(None, use_errno=True)
if sys.platform == "linux":
    _LIBC.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    _LIBC.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _LIBC.prctl.restype = ctypes.c_int


def _call(result, name):
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, name + ": " + os.strerror(code))


def mount(source, target, filesystem=None, flags=0, data=None):
    encoded = lambda value: value.encode() if value is not None else None
    _call(_LIBC.mount(encoded(source), encoded(target), encoded(filesystem), flags, encoded(data)), "mount " + target)


def readonly_bind(source, target):
    mount(source, target, flags=4096)
    mount(None, target, flags=4096 | 32 | 1 | 2 | 4)


def drop_authority():
    # Setup caps must not survive into the candidate or trusted collector.
    _call(_LIBC.prctl(38, 1, 0, 0, 0), "no_new_privs")
    _call(_LIBC.prctl(4, 0, 0, 0, 0), "disable process dumping")
    for capability in range(64):
        result = _LIBC.prctl(24, capability, 0, 0, 0)
        if result and ctypes.get_errno() != errno.EINVAL:
            _call(result, "drop capability bounding set")
    result = _LIBC.prctl(47, 4, 0, 0, 0)
    if result and ctypes.get_errno() != errno.EINVAL:
        _call(result, "clear ambient capabilities")
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    _call(_LIBC.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref((Data * 2)())), "clear process capabilities")


def _limited_child(command, timeout_s, max_log_bytes):
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   close_fds=True, start_new_session=True)
        failure = None
        while process.poll() is None:
            if time.monotonic() - started > timeout_s:
                failure = "timeout"
            elif os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > max_log_bytes:
                failure = "log_limit_exceeded"
            if failure:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.025)
        process.wait()
        if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > max_log_bytes:
            failure = failure or "log_limit_exceeded"
        stdout.seek(0)
        stderr.seek(0)
        return {"returncode": process.returncode, "stdout": stdout.read(max_log_bytes).decode(errors="replace"),
                "stderr": stderr.read(max_log_bytes).decode(errors="replace"), "failure_kind": failure,
                "elapsed_seconds": time.monotonic() - started}


_OBSERVE = r'''
import ctypes,fcntl,json,os,resource,socket
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True)
class Header(ctypes.Structure):_fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
class Data(ctypes.Structure):_fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
caps=(Data*2)()
if libc.capget(ctypes.byref(Header(0x20080522,0)),ctypes.byref(caps))!=0:raise RuntimeError('capget failed')
status={name:format(getattr(caps[0],field)|(getattr(caps[1],field)<<32),'016x') for name,field in [('CapInh','inheritable'),('CapPrm','permitted'),('CapEff','effective')]}
status['CapBnd']=format(sum((1<<i) for i in range(64) if libc.prctl(23,i,0,0,0)==1),'016x')
status['CapAmb']=format(sum((1<<i) for i in range(64) if libc.prctl(47,1,i,0,0)==1),'016x')
status['NoNewPrivs']=str(libc.prctl(39,0,0,0,0))
r={'protocol':'cpu-namespace-noproc/1','uid':os.geteuid(),'status':status,'proc_entries':[p.name for p in Path('/proc').iterdir()],'readonly':{},'writes':{},'affinity':sorted(os.sched_getaffinity(0)),'limits':{name:list(resource.getrlimit(getattr(resource,name))) for name in ['RLIMIT_AS','RLIMIT_CPU','RLIMIT_NPROC','RLIMIT_NOFILE','RLIMIT_FSIZE','RLIMIT_CORE']},'interfaces':list(socket.if_nameindex()),'api_env_present':any(('API' in k and 'KEY' in k) or k in ['SSH_AUTH_SOCK','AWS_SECRET_ACCESS_KEY','OPENAI_BASE_URL'] for k in os.environ)}
r['extra_descriptors']=[]
for descriptor in range(3,resource.getrlimit(resource.RLIMIT_NOFILE)[0]):
 try:fcntl.fcntl(descriptor,fcntl.F_GETFD);r['extra_descriptors'].append(descriptor)
 except OSError:pass
r['filesystem_bytes']={name:os.statvfs(name).f_frsize*os.statvfs(name).f_blocks for name in ['/tmp','/output']}
for name in ['/','/input','/candidate','/runner','/output','/tmp','/proc']:
 r['readonly'][name]=bool(os.statvfs(name).f_flag&os.ST_RDONLY)
for name in ['/.hf_readonly_marker','/input/.readonly_marker','/candidate/.readonly_marker','/output/.write_probe','/tmp/.write_probe']:
 try:Path(name).write_text('probe');r['writes'][name]='succeeded'
 except OSError as e:r['writes'][name]=type(e).__name__
for name in ['/output/.write_probe','/tmp/.write_probe']:
 try:Path(name).unlink()
 except OSError:pass
r['host_paths_visible']={}
for name in ['/var/run/docker.sock','/run/docker.sock','/reference','/verifier','/home/tione/notebook','/root/secrets']:
 try:r['host_paths_visible'][name]=Path(name).exists()
 except OSError:r['host_paths_visible'][name]=False
r['parent_root_accessible']=False
r['parent_fds_accessible']=False
try:os.readlink('/proc/1/root');r['parent_root_accessible']=True
except OSError:pass
try:list(Path('/proc/1/fd').iterdir());r['parent_fds_accessible']=True
except OSError:pass
s=socket.socket();s.settimeout(1)
try:s.connect(('1.1.1.1',443));r['network_connection']='succeeded'
except OSError as e:r['network_connection']=type(e).__name__
finally:s.close()
print(json.dumps(r))
'''


def _validate_observation(observed, config):
    reasons = []
    required = {"user", "mnt", "net", "pid", "ipc", "uts"}
    if set(config.get("host_namespaces", {})) != required or set(observed.get("namespaces", {})) != required:
        return ["complete host and container namespace identity is required"]
    for name, original in config["host_namespaces"].items():
        if observed["namespaces"].get(name) == original:
            reasons.append("namespace was not isolated: " + name)
    if observed["uid"] == 0:
        reasons.append("candidate uid is root")
    for name in ["CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"]:
        if int(observed["status"].get(name, "1"), 16) != 0:
            reasons.append("capability set was not cleared: " + name)
    if observed["status"].get("NoNewPrivs") != "1":
        reasons.append("no_new_privs was not set")
    if not all(observed["readonly"].get(name) for name in ["/", "/input", "/candidate", "/runner", "/proc"]):
        reasons.append("container root/input/runner/proc must be read-only")
    if any(observed["readonly"].get(name) for name in ["/output", "/tmp"]):
        reasons.append("bounded output/scratch tmpfs is not writable")
    for name in ["/.hf_readonly_marker", "/input/.readonly_marker", "/candidate/.readonly_marker"]:
        if observed["writes"].get(name) == "succeeded":
            reasons.append("write succeeded on a read-only path")
    if any(observed["host_paths_visible"].values()) or observed["api_env_present"]:
        reasons.append("host path, reference or secret environment is visible")
    if observed["parent_root_accessible"] or observed["parent_fds_accessible"]:
        reasons.append("candidate can inspect the trusted collector root or descriptors")
    if observed.get("extra_descriptors") != []:
        reasons.append("image process inherited unexpected descriptors")
    if any(name != "lo" for _, name in observed["interfaces"]) or observed["network_connection"] == "succeeded":
        reasons.append("private CPU network namespace is not disconnected")
    if observed.get("protocol") != "cpu-namespace-noproc/1" or observed.get("proc_entries") != []:
        reasons.append("explicit no-proc protocol requires an empty read-only /proc")
    if observed["limits"]["RLIMIT_AS"] != [config["memory_bytes"]] * 2:
        reasons.append("per-process address-space limit differs")
    expected_limits = {"RLIMIT_NPROC": config["pids_limit"], "RLIMIT_FSIZE": config["max_artifact_bytes"],
                       "RLIMIT_NOFILE": 128, "RLIMIT_CPU": math.ceil(config["timeout_s"]) + 5, "RLIMIT_CORE": 0}
    if any(observed["limits"].get(name) != [value] * 2 for name, value in expected_limits.items()):
        reasons.append("actual kernel resource limits differ from the explicit protocol")
    if observed["filesystem_bytes"].get("/output", config["max_artifact_bytes"] + 1) > config["max_artifact_bytes"]:
        reasons.append("aggregate output tmpfs is larger than the configured budget")
    if observed["filesystem_bytes"].get("/tmp", 256 * 1024**2 + 1) > 256 * 1024**2:
        reasons.append("scratch tmpfs is larger than the configured budget")
    if observed["affinity"] != config["affinity"]:
        reasons.append("CPU affinity differs")
    return reasons


def run(config):
    root, stage = Path(config["rootfs"]), Path(config["stage"])
    if sys.platform != "linux":
        raise RuntimeError("CPU namespace bootstrap requires Linux")
    if os.geteuid() == 0 or config.get("backend") != "numpy":
        raise RuntimeError("namespace launcher permits non-root CPU execution only")
    required = {"user", "mnt", "net", "pid", "ipc", "uts"}
    if config.get("protocol") != "cpu-namespace-noproc/1":
        raise RuntimeError("explicit cpu-namespace-noproc/1 protocol is required")
    if os.getpid() != 1 or set(config.get("host_namespaces", {})) != required:
        raise RuntimeError("trusted collector must be PID 1 in complete private CPU namespaces")
    namespace_snapshot = {name: os.readlink("/proc/self/ns/" + name) for name in required}
    if any(namespace_snapshot[name] == original for name, original in config["host_namespaces"].items()):
        raise RuntimeError("trusted collector namespace matches the host")
    inherited_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if inherited_limit < 0:
        inherited_limit = 1048576
    mount(None, "/", flags=(1 << 18) | 16384)  # private, recursive propagation
    mount(str(root), str(root), flags=4096)
    for name in ["input", "candidate", "runner"]:
        readonly_bind(str(stage / name), str(root / name))
    mount("tmpfs", str(root / "tmp"), "tmpfs", flags=2 | 4 | 8,
          data="size=256m,mode=1777")
    mount("tmpfs", str(root / "output"), "tmpfs", flags=2 | 4 | 8,
          data=f"size={config['max_artifact_bytes']},mode=1777")
    mount("tmpfs", str(root / "dev"), "tmpfs", flags=2 | 8, data="size=1m,mode=755")
    for name in ["null", "zero", "random", "urandom"]:
        (root / "dev" / name).touch()
        mount("/dev/" + name, str(root / "dev" / name), flags=4096)
    if list((root / "proc").iterdir()):
        raise RuntimeError("no-proc protocol refuses an existing proc tree")
    os.chdir(root)
    # pivot_root is architecture-specific; this executor qualifies x86_64 only.
    _call(_LIBC.syscall(155, ctypes.c_char_p(b"."), ctypes.c_char_p(b".old_root")), "pivot_root")
    os.chdir("/")
    _call(_LIBC.umount2(b"/.old_root", 2), "detach host root")
    mount(None, "/", flags=4096 | 32 | 1 | 2 | 4)
    for name, value in [(resource.RLIMIT_AS, config["memory_bytes"]),
                        (resource.RLIMIT_CPU, math.ceil(config["timeout_s"]) + 5),
                        (resource.RLIMIT_NPROC, config["pids_limit"]),
                        (resource.RLIMIT_NOFILE, 128), (resource.RLIMIT_FSIZE, config["max_artifact_bytes"]),
                        (resource.RLIMIT_CORE, 0)]:
        resource.setrlimit(name, (value, value))
    os.sched_setaffinity(0, config["affinity"])
    # Nothing outside the new root may remain reachable through descriptors.
    os.closerange(3, inherited_limit)
    os.environ.clear()
    os.environ.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    drop_authority()
    observation_process = _limited_child(["/usr/local/bin/python3", "-I", "-c", _OBSERVE], 15, config["max_log_bytes"])
    if observation_process["returncode"] != 0:
        raise RuntimeError("namespace isolation observation failed: " + observation_process["stderr"])
    observed = json.loads(observation_process["stdout"])
    observed["namespaces"] = namespace_snapshot
    observed["host_namespaces"] = config["host_namespaces"]
    observed["bootstrap_pid"] = 1
    observed["namespace_observation_source"] = "trusted PID 1 bootstrap readlink before pivot; image child inherits all six namespaces"
    observed["descriptor_policy"] = "all descriptors >=3 closed before image child; no host proc exposed"
    reasons = _validate_observation(observed, config)
    if reasons:
        return {"status": "invalid_runtime", "reasons": reasons, "isolation": observed}
    package_process = _limited_child(["/usr/local/bin/python3", "-I", "-c", config["probe_code"], "numpy"], 30, config["max_log_bytes"])
    if package_process["returncode"] != 0:
        raise RuntimeError("CPU image dependency probe failed: " + package_process["stderr"])
    probe = json.loads(package_process["stdout"])
    receipt = {"status": "passed", "probe": probe, "isolation": observed, "artifacts_base64": {}}
    if config["mode"] == "worker":
        command = ["/usr/local/bin/python3", "-I", "/runner/worker.py", "--candidate", "/candidate/solution.py",
                   "--problem", "/input/problem.npz", "--output", "/output/result.npz", "--backend", "numpy",
                   "--options", "/input/options.json"]
        process = _limited_child(command, config["timeout_s"], config["max_log_bytes"])
        # This process is PID 1 in a private namespace. Kill remaining workers
        # before collecting shared-UID files, so descendants cannot race reads.
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process["status"] = "passed" if process["returncode"] == 0 and not process["failure_kind"] else "failed"
        receipt.update(status=process["status"], process=process)
        total = 0
        for path in Path("/output").iterdir():
            info = path.lstat()
            if path.name not in {"result.npz", "worker.json"} or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("unexpected or linked namespace output artifact")
            total += info.st_size
            if total > config["max_artifact_bytes"]:
                raise RuntimeError("namespace output exceeds aggregate tmpfs budget")
            receipt["artifacts_base64"][path.name] = base64.b64encode(path.read_bytes()).decode()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        receipt = run(config)
    except Exception as exc:
        receipt = {"status": "invalid_runtime", "reasons": [type(exc).__name__ + ": " + str(exc)]}
    print("HF_NAMESPACE_RESULT=" + json.dumps(receipt, allow_nan=False, separators=(",", ":")), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
