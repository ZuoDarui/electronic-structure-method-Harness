"""CPU namespace protocol controls; no GPU, remote login or namespace creation."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from harness_engineering.generation.namespace_executor import NamespaceExecutor
from harness_engineering.generation.namespace_launcher import _validate_observation, run as launch
from test_generation_executor import SOLUTION, problem_file, process


MODULE = 'harness_engineering.generation.namespace_executor'
CONFIG = {'ssh_host':'operator@example.org','namespace_protocol':'cpu-namespace-noproc/1','namespace_rootfs':'/trusted/rootfs',
          'namespace_rootfs_manifest':'/trusted/rootfs.manifest.json',
          'namespace_rootfs_manifest_sha256':'a'*64,'cpu_image':'sha256:'+'b'*64}


def observations():
    host={name:name+':host' for name in ['user','mnt','net','pid','ipc','uts']}
    config={'host_namespaces':host,'memory_bytes':2*1024**3,'affinity':[0,1],
            'pids_limit':64,'max_artifact_bytes':256*1024**2,'timeout_s':300}
    observed={'namespaces':{name:name+':container' for name in host},'uid':5033,
              'status':{name:'0000000000000000' for name in ['CapInh','CapPrm','CapEff','CapBnd','CapAmb']},
              'readonly':{name:True for name in ['/','/input','/candidate','/runner','/proc']},
              'writes':{name:'OSError' for name in ['/.hf_readonly_marker','/input/.readonly_marker','/candidate/.readonly_marker']},
              'host_paths_visible':{'/var/run/docker.sock':False,'/reference':False},
              'api_env_present':False,'parent_root_accessible':False,'parent_fds_accessible':False,
              'interfaces':[[1,'lo']],'network_connection':'OSError','protocol':'cpu-namespace-noproc/1','proc_entries':[],
              'limits':{'RLIMIT_AS':[2*1024**3]*2,'RLIMIT_NPROC':[64,64],'RLIMIT_FSIZE':[256*1024**2]*2,
                        'RLIMIT_NOFILE':[128,128],'RLIMIT_CPU':[305,305],'RLIMIT_CORE':[0,0]},
              'extra_descriptors':[],'filesystem_bytes':{'/tmp':256*1024**2,'/output':256*1024**2},'affinity':[0,1]}
    observed['status']['NoNewPrivs']='1'
    observed['readonly'].update({'/output':False,'/tmp':False})
    return config,observed


class NamespaceAdmissionTests(unittest.TestCase):
    def test_all_six_namespaces_and_capability_sets_are_required(self):
        config,observed=observations()
        self.assertEqual(_validate_observation(observed,config),[])
        for name in config['host_namespaces']:
            with self.subTest(name=name):
                changed=copy.deepcopy(observed);changed['namespaces'][name]=config['host_namespaces'][name]
                self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);del changed['namespaces']['pid']
        self.assertTrue(_validate_observation(changed,config))
        for name in ['CapInh','CapPrm','CapEff','CapBnd','CapAmb']:
            with self.subTest(capability=name):
                changed=copy.deepcopy(observed);changed['status'][name]='1'
                self.assertTrue(_validate_observation(changed,config))

    def test_secret_paths_parent_descriptors_or_network_visibility_reject(self):
        config,observed=observations()
        for name in ['api_env_present','parent_root_accessible','parent_fds_accessible']:
            with self.subTest(name=name):
                changed=copy.deepcopy(observed);changed[name]=True
                self.assertTrue(_validate_observation(changed,config))
        for change in [('host_paths_visible',{'/reference':True}),('interfaces',[[1,'lo'],[2,'eth0']]),('network_connection','succeeded')]:
            changed=copy.deepcopy(observed);changed[change[0]]=change[1]
            self.assertTrue(_validate_observation(changed,config))

    def test_root_user_mutable_input_or_wrong_resource_limits_reject(self):
        config,observed=observations()
        changed=copy.deepcopy(observed);changed['uid']=0
        self.assertTrue(_validate_observation(changed,config))
        for name in ['/','/input','/candidate','/runner','/proc']:
            changed=copy.deepcopy(observed);changed['readonly'][name]=False
            self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);changed['limits']['RLIMIT_AS']=[-1,-1]
        self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);changed['affinity']=[0,1,2]
        self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);changed['filesystem_bytes']['/output']*=2
        self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);changed['extra_descriptors']=[8]
        self.assertTrue(_validate_observation(changed,config))
        changed=copy.deepcopy(observed);changed['proc_entries']=['1']
        self.assertTrue(_validate_observation(changed,config))

    def test_cpu_only_admission_never_calls_remote_for_gpu(self):
        executor=NamespaceExecutor(CONFIG)
        with patch.object(executor,'_launch') as launch_remote:
            self.assertEqual(executor.preflight('cupy')['status'],'invalid_runtime')
            self.assertEqual(executor.run(Path('/missing'),Path('/missing'),Path('/missing'),'cupy')['status'],'invalid_runtime')
            launch_remote.assert_not_called()

    def test_runtime_paths_and_manifest_must_be_explicit(self):
        for change in [{'namespace_protocol':None},{'namespace_rootfs':'relative'},{'namespace_rootfs_manifest_sha256':'tag'},{'namespace_rootfs_manifest':None}]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                NamespaceExecutor({**CONFIG,**change})

    def test_probe_identity_excludes_dynamic_namespace_observation(self):
        executor=NamespaceExecutor(CONFIG)
        identity={'id':CONFIG['cpu_image'],'rootfs_manifest_sha256':'a'*64}
        probe={'uid':5033,'numpy':'1.26.4','loaded_modules':{'numpy':{'sha256':'c'*64}}}
        replies=[{'status':'passed','image':identity,'probe':probe,'isolation':{'inode':1}},
                 {'status':'passed','image':identity,'probe':probe,'isolation':{'inode':2}}]
        with patch.object(executor,'_launch',side_effect=replies):
            first,second=executor.preflight('numpy'),executor.preflight('numpy')
        self.assertEqual(first['image'],second['image'])
        self.assertEqual(first['probe'],second['probe'])
        self.assertNotEqual(first['isolation'],second['isolation'])

    def test_transport_stage_never_contains_private_references(self):
        executor=NamespaceExecutor(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['input','candidate','runner','output']:(root/name).mkdir()
            (root/'candidate/solution.py').write_text(SOLUTION);problem_file(root/'input/problem.npz')
            (root/'input/options.json').write_text('{}');(root/'runner/worker.py').write_text('trusted public worker')
            def transport(command,timeout_s,max_log_bytes,input_data=None):
                with tarfile.open(fileobj=io.BytesIO(input_data),mode='r:') as archive:
                    self.assertEqual(set(archive.getnames()),{'candidate/solution.py','input/problem.npz','input/options.json','runner/worker.py','runner/namespace_launcher.py','namespace-plan.json'})
                self.assertIn('--map-current-user',command[-1])
                self.assertIn('os.environ.clear()',command[-1])
                return process(json.dumps({'status':'passed','image':{},'probe':{}}))
            with patch(MODULE+'._bounded_process',side_effect=transport):
                self.assertEqual(executor._launch(root,'worker')['status'],'passed')

    def test_isolation_failure_remains_invalid_runtime_not_numerical_retry(self):
        executor=NamespaceExecutor(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);candidate=root/'candidate';candidate.mkdir()
            (candidate/'solution.py').write_text(SOLUTION);problem_file(root/'problem.npz')
            runtime={'status':'passed','backend':'numpy','require_external_trace':False,'image':{},'probe':{}}
            with patch.object(executor,'preflight',return_value=runtime),patch.object(executor,'_launch',return_value={'status':'invalid_runtime','reasons':['private mount unavailable']}):
                result=executor.run(candidate,root/'problem.npz',root/'output','numpy')
            self.assertEqual(result['status'],'invalid_runtime')
            self.assertEqual(json.loads((root/'output/execution.json').read_text())['status'],'invalid_runtime')

    def test_actual_worker_image_or_dependency_drift_is_invalid_runtime(self):
        executor=NamespaceExecutor(CONFIG)
        admitted={'image':{'id':'frozen-image'},'probe':{'numpy':'frozen'}}
        with patch.object(executor,'_launch',return_value={'status':'passed','image':{'id':'other'},'probe':{'numpy':'frozen'}}):
            process=executor._execute_remote(Path('/unused'),'numpy',admitted)[0]
        self.assertEqual(process['failure_kind'],'isolation_policy')
        self.assertIsNone(process['returncode'])


if __name__=='__main__':
    unittest.main()
