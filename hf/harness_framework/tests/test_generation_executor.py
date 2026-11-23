"""Isolation command/receipt tests and direct worker protocol controls.

Worker subprocess tests deliberately run on the host to test serialization only;
they do not claim Docker isolation. Docker behavior is mocked when no daemon is
available. A missing real daemon must remain invalid_runtime.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from harness_engineering.generation.executor import DockerExecutor, validate_external_trace, _bounded_process, _PROBE, check_container_policy, _checked_docker_process
from harness_engineering.generation.worker import STAGES


IMAGE = "sha256:" + "a" * 64
MODULE = "harness_engineering.generation.executor"
WORKER = Path(__file__).resolve().parents[1] / "harness_engineering/generation/worker.py"
SOLUTION = '''import numpy as np
def build_jk(problem, density, backend):
    assert backend == 'numpy'
    assert 'metadata' in problem and 'metadata_json' not in problem
    assert not any('reference' in name for name in problem)
    eri = problem['eri']
    return (np.einsum('pqrs,rs->pq', eri, density),
            np.einsum('prqs,rs->pq', eri, density))
def solve(problem, options, backend):
    energy, coeff = np.linalg.eigh(problem['hcore'])
    occupations = np.asarray([2.,0.], dtype=np.float64)
    density = (coeff*occupations)@coeff.T
    total = float(np.sum(density*problem['hcore'])+problem['enuc'])
    return {'energy':total,'density':density,'mo_coeff':coeff,'mo_energy':energy,
            'occupations':occupations,'converged':True,
            'history':[{'cycle':1,'energy':total,'residual_norm':0.0}]}
'''


def problem_file(path):
    np.savez_compressed(path, overlap=np.eye(2), hcore=np.diag([-1., -.2]),
                        enuc=np.asarray(.5), nelectron=np.asarray(2, dtype=np.int64),
                        eri=np.zeros((2, 2, 2, 2)), trial_densities=np.array([np.eye(2), np.eye(2)*.25]),
                        metadata_json=np.asarray(json.dumps({"id": "toy", "method": "RHF"})))


def process(stdout="", stderr="", returncode=0):
    return {"returncode": returncode, "stdout": stdout, "stderr": stderr,
            "failure_kind": None, "elapsed_seconds": .01}


def runtime(backend="numpy", external=False):
    return {"status": "passed", "backend": backend,
            "require_external_trace": external,
            "image": {"id": IMAGE, "os": "linux", "architecture": "amd64", "repo_digests": []}}


def actual_policy():
    return {'Image': IMAGE, 'Config': {'User':'65532:65532'}, 'Mounts': [],
            'HostConfig': {'NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,
                           'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges:true'],'Init':True,
                           'PidsLimit':64,'Memory':2048*1024**2,'MemorySwap':2048*1024**2,
                           'NanoCpus':10**9,'Tmpfs':{'/tmp':'rw,nosuid,nodev,size=256m,mode=1777'}}}


def trace_database(path, missing=None, wrong_thread=None, outside=None):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER)")
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,demangledName INTEGER)")
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER)")
        connection.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        connection.execute("INSERT INTO StringIds VALUES(1,'matrix_kernel')")
        for index, stage in enumerate(STAGES):
            start = 1000 * (index + 1)
            connection.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)", (start, start + 500, 99, stage))
            if stage == missing:
                continue
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)", (start+10, start+20, 5 if stage == wrong_thread else 99, index+1))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,1)", (start+30, start+900 if stage == outside else start+80, index+1))


class ExternalTraceTests(unittest.TestCase):
    def test_requires_correlated_kernel_in_each_trusted_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"trace.sqlite"
            trace_database(path)
            evidence = validate_external_trace(path)
            self.assertTrue(evidence["valid"], evidence)
            self.assertEqual(set(evidence["stage_coverage"]), set(STAGES))
            self.assertTrue(all(c["kernel_count"] == 1 for c in evidence["stage_coverage"].values()))

    def test_import_probe_or_only_final_gpu_work_does_not_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"trace.sqlite"
            trace_database(path, missing="hf.jk1")
            evidence = validate_external_trace(path)
            self.assertFalse(evidence["valid"])
            self.assertIn("no correlated CUDA kernel inside hf.jk1", evidence["reasons"])

    def test_kernel_must_belong_to_stage_thread_and_interval(self):
        for kwargs in ({"wrong_thread": "hf.solve"}, {"outside": "hf.solve"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"trace.sqlite"
                trace_database(path, **kwargs)
                self.assertFalse(validate_external_trace(path)["valid"])

    def test_missing_malformed_or_symlinked_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(validate_external_trace(root/"missing")["valid"])
            (root/"bad").write_text("not sqlite")
            self.assertFalse(validate_external_trace(root/"bad")["valid"])
            (root/"link").symlink_to(root/"bad")
            self.assertFalse(validate_external_trace(root/"link")["valid"])


class DockerExecutorTests(unittest.TestCase):
    def test_probe_hashes_actual_imported_modules_without_volatile_identity(self):
        # Protocol-only host subprocess: disable only the image's no-PySCF
        # admission check because this trusted test environment has PySCF.
        prefix = ("import importlib.util,sys; original=importlib.util.find_spec; "
                  "importlib.util.find_spec=lambda name,*a,**kw: None if name=='pyscf' else original(name,*a,**kw); "
                  "sys.argv=['probe','numpy']; ")
        command = [sys.executable, '-c', prefix + 'exec(' + repr(_PROBE) + ')']
        first = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
        second = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
        identity = json.loads(first.stdout)
        self.assertEqual(identity, json.loads(second.stdout))
        self.assertEqual(set(identity['loaded_modules']),
                         {'numpy','scipy','numpy._core._multiarray_umath','scipy.linalg._flapack'})
        for record in identity['loaded_modules'].values():
            self.assertTrue(Path(record['path']).is_file())
            self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())
        self.assertNotIn('pid', identity)

    def test_missing_docker_is_invalid_runtime_without_host_fallback(self):
        executor = DockerExecutor({"docker_command": ["/nonexistent/hf-no-docker"]})
        evidence = executor.preflight("numpy")
        self.assertEqual(evidence["status"], "invalid_runtime")
        self.assertIn("unavailable", " ".join(evidence["reasons"]))

    def test_preflight_pins_actual_linux_image_id(self):
        replies = [process(json.dumps({"Os": "linux", "Version": "27.5", "Arch": "amd64"})),
                   process(json.dumps(runtime()["image"]))]
        probe={**process(json.dumps({"uid":65532,"numpy":"2.2.6","pyscf_available":False})),
               'isolation_policy':{'valid':True,'reasons':[]}}
        with patch(MODULE+"._bounded_process", side_effect=replies), patch(MODULE+'._checked_docker_process',return_value=probe) as mocked:
            evidence = DockerExecutor({"cpu_image": "mutable:tag"}).preflight("numpy")
        self.assertEqual(evidence["status"], "passed", evidence)
        self.assertEqual(evidence["image"]["id"], IMAGE)
        self.assertIn(IMAGE, mocked.call_args.args[0])

    def test_non_linux_daemon_is_refused(self):
        with patch(MODULE+"._bounded_process", return_value=process('{"Os":"windows"}')):
            evidence = DockerExecutor().preflight("numpy")
        self.assertEqual(evidence["status"], "invalid_runtime")

    def test_gpu_requires_external_trace_by_default(self):
        replies = [process('{"Os":"linux"}'), process(json.dumps(runtime()["image"]))]
        probes=[{**process('{"uid":65532,"cupy":"13.6.0"}'),'isolation_policy':{'valid':True}},
                process(stderr="nsys not found",returncode=127)]
        with patch(MODULE+"._bounded_process", side_effect=replies), patch(MODULE+'._checked_docker_process',side_effect=probes):
            evidence = DockerExecutor().preflight("cupy")
        self.assertEqual(evidence["status"], "invalid_runtime")
        self.assertTrue(evidence["require_external_trace"])
        self.assertIn("requires nsys", " ".join(evidence["reasons"]))

    def test_actual_daemon_policy_must_match_before_payload_start(self):
        executor=DockerExecutor()
        command=executor.docker+executor._flags('numpy','hf-check')+[IMAGE,'python3','-c','pass']
        valid=actual_policy()
        self.assertTrue(check_container_policy(valid,command[1:])['valid'])
        bad=actual_policy();bad['HostConfig'].update(NetworkMode='host',ReadonlyRootfs=False,CapDrop=None,SecurityOpt=None,PidsLimit=None,Tmpfs=None)
        replies=[process('created'),process(json.dumps([bad])),process()]
        with patch(MODULE+'._bounded_process',side_effect=replies) as mocked:
            evidence=_checked_docker_process(command,executor.docker,10,65536)
        self.assertEqual(evidence['failure_kind'],'isolation_policy')
        self.assertFalse(evidence['isolation_policy']['valid'])
        self.assertFalse(any('start' in call.args[0] for call in mocked.call_args_list))
        self.assertTrue(any('NetworkMode' in reason for reason in evidence['isolation_policy']['reasons']))

    def test_each_requested_readonly_mount_must_be_preserved(self):
        executor=DockerExecutor()
        flags=executor._flags('numpy','hf-check')+['--mount','type=bind,src=/tmp/input,dst=/input,readonly',IMAGE]
        raw=actual_policy();raw['Mounts']=[{'Type':'bind','Source':'/tmp/input','Destination':'/input','RW':False}]
        self.assertTrue(check_container_policy(raw,flags)['valid'])
        raw['Mounts'][0]['RW']=True
        self.assertFalse(check_container_policy(raw,flags)['valid'])

    def test_checked_start_uses_actual_container_exit_code(self):
        executor=DockerExecutor();command=executor.docker+executor._flags('numpy','hf-check')+[IMAGE,'python3','-c','pass']
        replies=[process('created'),process(json.dumps([actual_policy()])),process('payload stdout'),
                 process(json.dumps([{'State':{'Running':False,'ExitCode':7}}])),process()]
        with patch(MODULE+'._bounded_process',side_effect=replies) as mocked:
            evidence=_checked_docker_process(command,executor.docker,10,65536)
        self.assertEqual(evidence['returncode'],7)
        self.assertEqual(evidence['stdout'],'payload stdout')
        self.assertTrue(any('start' in call.args[0] for call in mocked.call_args_list))

    def test_security_flags_and_mounts_do_not_include_oracle_or_secret(self):
        executor = DockerExecutor()
        commands, _ = executor._commands(Path('/tmp/stage'), 'numpy', runtime())
        command = commands[0]
        for flag in ('--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges:true','--pull=never'):
            self.assertIn(flag, command)
        self.assertIn('65532:65532', command)
        mounts = [command[i+1] for i,value in enumerate(command[:-1]) if value == '--mount']
        self.assertEqual(len(mounts), 4)
        self.assertEqual(sum('readonly' in value for value in mounts), 3)
        self.assertFalse(any('reference' in value or 'secret' in value for value in mounts))
        self.assertNotIn('--privileged', command)
        self.assertNotIn('--env-file', command)

    def test_external_profile_collects_cuda_and_nvtx_then_exports_sqlite(self):
        commands, _ = DockerExecutor()._commands(Path('/tmp/stage'), 'cupy', runtime('cupy', True))
        self.assertEqual(len(commands), 2)
        self.assertIn('--trace=cuda,nvtx', commands[0])
        self.assertIn('--sample=none', commands[0])
        self.assertIn('--type=sqlite', commands[1])
        self.assertIn('/output/cuda_trace.nsys-rep', commands[1])

    def test_only_solution_is_staged_and_hash_identity_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root/'candidate', root/'out'
            source.mkdir()
            (source/'solution.py').write_text(SOLUTION)
            (source/'private-reference.txt').write_text('must never be mounted')
            problem_file(root/'problem.npz')
            executor = DockerExecutor()
            def execute(stage, commands, names):
                self.assertEqual([p.name for p in (stage/'candidate').iterdir()], ['solution.py'])
                np.savez_compressed(stage/'output/result.npz', energy=np.asarray(-1.5))
                record = {'status':'passed','candidate_sha256':hashlib.sha256((source/'solution.py').read_bytes()).hexdigest(),
                          'input_sha256':hashlib.sha256((root/'problem.npz').read_bytes()).hexdigest(),
                          'output_sha256':hashlib.sha256((stage/'output/result.npz').read_bytes()).hexdigest()}
                (stage/'output/worker.json').write_text(json.dumps(record))
                return [process()]
            with patch.object(executor, 'preflight', return_value=runtime()), patch.object(executor, '_execute_local', side_effect=execute):
                evidence = executor.run(source, root/'problem.npz', out, 'numpy', {'max_cycle':100})
            self.assertEqual(evidence['status'], 'passed', evidence)
            self.assertEqual(evidence['input_sha256'], hashlib.sha256((root/'problem.npz').read_bytes()).hexdigest())
            self.assertTrue(Path(evidence['output_path']).is_file())
            self.assertTrue((out/'execution.json').is_file())

    def test_stale_output_directory_and_symlinked_candidate_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'candidate';source.mkdir();problem_file(root/'problem.npz')
            (root/'real.py').write_text(SOLUTION);(source/'solution.py').symlink_to(root/'real.py')
            executor=DockerExecutor()
            with patch.object(executor,'preflight') as preflight:
                self.assertEqual(executor.run(source,root/'problem.npz',root/'out')['status'],'invalid_runtime')
                preflight.assert_not_called()
            (source/'solution.py').unlink();(source/'solution.py').write_text(SOLUTION)
            (root/'out').mkdir();(root/'out/result.npz').write_bytes(b'stale')
            with patch.object(executor,'preflight') as preflight:
                evidence=executor.run(source,root/'problem.npz',root/'out')
                self.assertEqual(evidence['status'],'invalid_runtime')
                self.assertIn('stale results',' '.join(evidence['reasons']))
                preflight.assert_not_called()

    def test_ssh_transport_uses_batch_and_strict_host_identity(self):
        executor = DockerExecutor({'ssh_host':'root@example.org','ssh_port':11092,
                                   'ssh_options':['-o','UserKnownHostsFile=/tmp/hf-known-hosts']})
        with patch(MODULE+'._bounded_process', return_value=process('{}')) as mocked:
            executor._docker(['version'])
        command=mocked.call_args.args[0]
        self.assertIn('BatchMode=yes',command)
        self.assertIn('StrictHostKeyChecking=yes',command)
        self.assertIn('11092',command)
        self.assertEqual(command[-1],'docker version')

    def test_root_container_and_invalid_resource_limits_are_rejected(self):
        for config in ({'user':'0:0'},{'memory_mb':float('nan')},{'timeout_s':True},
                       {'pids_limit':.5},{'require_external_trace':'false'},
                       {'ssh_host':'-oProxyCommand=bad'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DockerExecutor(config)

    def test_strict_gpu_requires_external_trace_and_observed_eigensolver(self):
        for eigen_count in (0, 1):
            with self.subTest(eigen_count=eigen_count), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'candidate';source.mkdir()
                (source/'solution.py').write_text(SOLUTION);problem_file(root/'problem.npz')
                executor=DockerExecutor()
                def execute(stage,commands,names):
                    np.savez_compressed(stage/'output/result.npz',energy=np.asarray(-1.5))
                    trace_database(stage/'output/cuda_trace.sqlite')
                    (stage/'output/cuda_trace.nsys-rep').write_bytes(b'external collector artifact')
                    record={'status':'passed','candidate_sha256':hashlib.sha256((source/'solution.py').read_bytes()).hexdigest(),
                            'input_sha256':hashlib.sha256((root/'problem.npz').read_bytes()).hexdigest(),
                            'output_sha256':hashlib.sha256((stage/'output/result.npz').read_bytes()).hexdigest(),
                            'gpu_trace':{'stage_coverage':{'hf.solve':{'eigensolver_count':eigen_count}}}}
                    (stage/'output/worker.json').write_text(json.dumps(record))
                    return [process(),process()]
                with patch.object(executor,'preflight',return_value=runtime('cupy',True)), patch.object(executor,'_execute_local',side_effect=execute):
                    evidence=executor.run(source,root/'problem.npz',root/'out','cupy')
                self.assertTrue(evidence['trace']['external']['valid'])
                self.assertEqual(evidence['status'],'passed' if eigen_count else 'failed')
                self.assertIn('result.npz',evidence['hashes']['artifacts'])

    def test_numeric_crash_remains_failed_when_trace_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'candidate';source.mkdir()
            (source/'solution.py').write_text(SOLUTION);problem_file(root/'problem.npz')
            executor=DockerExecutor()
            with patch.object(executor,'preflight',return_value=runtime('cupy',True)), patch.object(executor,'_execute_local',return_value=[process(stderr='candidate error',returncode=1)]):
                evidence=executor.run(source,root/'problem.npz',root/'out','cupy')
            self.assertEqual(evidence['status'],'failed')
            self.assertIsNone(evidence['output_path'])

    def test_ssh_tar_contains_only_public_inputs_and_trusted_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)
            for name in ('input','candidate','runner','output'):(stage/name).mkdir()
            (stage/'candidate/solution.py').write_text(SOLUTION)
            (stage/'runner/worker.py').write_bytes(WORKER.read_bytes())
            problem_file(stage/'input/problem.npz');(stage/'input/options.json').write_text('{}')
            executor=DockerExecutor({'ssh_host':'root@example.org','remote_root':'/tmp/isolated-root'})
            def remote(command,timeout,max_log,input_data=None,binary=False):
                with tarfile.open(fileobj=io.BytesIO(input_data),mode='r:') as archive:
                    self.assertEqual(set(archive.getnames()),{'candidate/solution.py','runner/worker.py','input/problem.npz','input/options.json','execution-plan.json'})
                    plan=json.load(archive.extractfile('execution-plan.json'))
                    self.assertTrue(all('--network=none' in c for c in plan['commands']))
                self.assertIn("dir='/tmp/isolated-root'",shlex.split(command[-1])[-1])
                bundle=io.BytesIO()
                with tarfile.open(fileobj=bundle,mode='w') as archive:
                    data=json.dumps([process()]).encode();member=tarfile.TarInfo('process.json');member.size=len(data)
                    archive.addfile(member,io.BytesIO(data))
                return process(stdout=bundle.getvalue())
            with patch(MODULE+'._bounded_process',side_effect=remote):
                records=executor._execute_remote(stage,'numpy',runtime())
            self.assertEqual(records[0]['returncode'],0)

    def test_process_capture_and_timeout_are_bounded(self):
        evidence=_bounded_process([sys.executable,'-c','import time; time.sleep(10)'],.03,1024)
        self.assertEqual(evidence['failure_kind'],'timeout')
        evidence=_bounded_process([sys.executable,'-c',"import sys; sys.stdout.write('x'*100000); sys.stdout.flush()"],2,128)
        self.assertEqual(evidence['failure_kind'],'log_limit_exceeded')
        self.assertLessEqual(len(evidence['stdout'].encode()),128)


class WorkerProtocolTests(unittest.TestCase):
    def invoke(self, source=SOLUTION, mutate_problem=None):
        # This is intentionally a direct host subprocess protocol check.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'solution.py').write_text(source);problem_file(root/'problem.npz')
            if mutate_problem:
                mutate_problem(root/'problem.npz')
            (root/'options.json').write_text('{}')
            command=[sys.executable,str(WORKER),'--candidate',str(root/'solution.py'),'--problem',str(root/'problem.npz'),
                     '--output',str(root/'out/result.npz'),'--backend','numpy','--options',str(root/'options.json')]
            completed=subprocess.run(command,capture_output=True,text=True,timeout=20)
            record=json.loads((root/'out/worker.json').read_text())
            arrays=None
            if (root/'out/result.npz').is_file():
                with np.load(root/'out/result.npz',allow_pickle=False) as archive:
                    arrays={k:archive[k].copy() for k in archive.files}
            return completed,record,arrays

    def test_serializes_raw_jk_and_solver_arrays_without_self_reported_errors(self):
        completed,record,arrays=self.invoke()
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertEqual(record['status'],'passed')
        self.assertFalse(record['reference_available'])
        self.assertEqual(set(arrays),{'j0','k0','j1','k1','jsum','ksum','density','mo_coeff','mo_energy','occupations','energy','converged','history_json'})
        self.assertTrue(bool(arrays['converged']))
        self.assertEqual(arrays['converged'].dtype.kind,'b')
        self.assertNotIn('metrics',record)
        self.assertEqual(record['input_sha256'],record['input_sha256_after'])

    def test_nonfinite_or_float32_arrays_are_not_serialized(self):
        for replacement in ("density.astype(np.float32)", "density*np.nan"):
            with self.subTest(replacement=replacement):
                completed,record,arrays=self.invoke(SOLUTION.replace("'density':density", "'density':"+replacement))
                self.assertNotEqual(completed.returncode,0)
                self.assertEqual(record['status'],'failed')
                self.assertIsNone(arrays)

    def test_missing_or_inconsistent_history_is_rejected(self):
        source=SOLUTION.replace("'cycle':1", "'cycle':0")
        completed,record,arrays=self.invoke(source)
        self.assertNotEqual(completed.returncode,0)
        self.assertIn('history cycles',record['error'])
        self.assertIsNone(arrays)

    def test_unknown_input_fields_cannot_smuggle_a_reference(self):
        def inject(path):
            with np.load(path,allow_pickle=False) as archive:
                arrays={k:archive[k].copy() for k in archive.files}
            arrays['reference_density']=np.eye(2)
            np.savez_compressed(path,**arrays)
        completed,record,arrays=self.invoke(mutate_problem=inject)
        self.assertNotEqual(completed.returncode,0)
        self.assertIn('outside the candidate contract',record['error'])
        self.assertIsNone(arrays)


if __name__ == '__main__':
    unittest.main()
