"""Unified entry point for the generation and optimization harness."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest

FRAMEWORK_ROOT = Path(__file__).resolve().parent
HF_ROOT = FRAMEWORK_ROOT.parent
CPU_TEST_MODULES = (
    'test_generation_generator', 'test_generation_streaming',
    'test_generation_json_transport', 'test_generation_science',
    'test_generation_executor', 'test_generation_namespace',
    'test_generation_campaign', 'test_generation_campaign_review',
    'test_generation_single',
    'test_harness_engineering', 'test_framework_layout',
)


def cpu_tests(argv):
    parser = argparse.ArgumentParser(description='Explicit offline CPU harness regression; no model or GPU calls')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or FRAMEWORK_ROOT / 'outputs/tests' / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for path in (FRAMEWORK_ROOT / 'tests', HF_ROOT):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    protected = {name: hashlib.sha256((HF_ROOT / name).read_bytes()).hexdigest()
                 for name in ('hf.py', 'gpu_hf.py', 'make_alkane_xyz.py')}
    with (output / 'tests.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        loader = unittest.TestLoader()
        suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in CPU_TEST_MODULES)
        suite.addTests(loader.loadTestsFromName('test_direct_hf'))
        baseline = importlib.import_module('test_gpu_hf').GPUFriendlyRHFTests
        suite.addTests(baseline(name) for name in loader.getTestCaseNames(baseline)
                       if name != 'test_explicit_gpu_request_does_not_silently_run_on_cpu')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    intact = all(hashlib.sha256((HF_ROOT / name).read_bytes()).hexdigest() == sha for name, sha in protected.items())
    receipt = {
        'passed': result.wasSuccessful() and intact, 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
        'python': sys.executable, 'framework_root': str(FRAMEWORK_ROOT), 'hf_root': str(HF_ROOT),
        'outer_hf_sources_unchanged': intact, 'outer_source_sha256': protected,
        'model_requests': False, 'gpu_execution': False, 'automatic_test_discovery': False,
        'scope': 'Framework regression and existing CPU baselines; not new generated-method acceptance.',
    }
    (output / 'result.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({**receipt, 'output': str(output)}, ensure_ascii=False, indent=2))
    return 0 if receipt['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='Unified HF generation and optimization harness')
    subparsers = parser.add_subparsers(dest='operation', required=True)
    subparsers.add_parser('generate', add_help=False, help='Generate, resume or evaluate a method campaign')
    subparsers.add_parser('optimize', add_help=False, help='Run an existing optimization or correctness task')
    subparsers.add_parser('loop', add_help=False, help='Run the legacy oracle/budget decision workflow once')
    subparsers.add_parser('test', add_help=False, help='Run explicit offline CPU regression tests')
    args, remainder = parser.parse_known_args(argv)
    # Task commands, editable paths and default output paths share one root.
    os.chdir(FRAMEWORK_ROOT)
    if str(FRAMEWORK_ROOT) not in sys.path:
        sys.path.insert(0, str(FRAMEWORK_ROOT))
    from harness_engineering.paths import configure_import_paths
    configure_import_paths()
    if args.operation == 'generate':
        from harness_engineering.generation.__main__ import main as operation
    elif args.operation == 'optimize':
        from harness_engineering.cli import main as operation
    elif args.operation == 'loop':
        from harness_engineering.loop import main as operation
    else:
        operation = cpu_tests
    return operation(remainder)


if __name__ == '__main__':
    raise SystemExit(main())
