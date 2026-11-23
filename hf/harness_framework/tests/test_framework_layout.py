"""Exercise the relocated framework against the existing parent HF files.

Import probes never call an example main, a model, or a GPU backend. The one
trusted-reference fixture is a single helium atom calculated on the CPU.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from harness_engineering import paths
from harness_engineering.generation import __main__ as generation_entry
from harness_engineering.generation.science import prepare_case
from harness_engineering.loop import changed_files, snapshot_files


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = FRAMEWORK_ROOT.parent


class FrameworkLayoutTests(unittest.TestCase):
    def probe(self, source, *arguments, cwd):
        environment = {key: os.environ[key] for key in (
            "PATH", "SYSTEMROOT", "TMPDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"
        ) if key in os.environ}
        environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        return subprocess.run([sys.executable, "-I", "-c", source, *map(str, arguments)],
                              cwd=cwd, env=environment, text=True,
                              capture_output=True, timeout=60, check=False)

    def test_resources_and_existing_algorithms_have_distinct_roots(self):
        self.assertEqual(paths.FRAMEWORK_ROOT, FRAMEWORK_ROOT)
        self.assertEqual(paths.HF_ROOT, HF_ROOT)
        for resource in ("harness/tasks/rhf_direct_baseline.json",
                         "harness/generation/rhf.json", "containers/Dockerfile.cpu"):
            self.assertTrue((FRAMEWORK_ROOT / resource).is_file(), resource)
        for name in ("hf.py", "gpu_hf.py", "make_alkane_xyz.py"):
            self.assertTrue((HF_ROOT / name).is_file(), name)
            self.assertFalse((FRAMEWORK_ROOT / name).exists(), name)

    def test_all_direct_examples_import_parent_algorithms_and_geometry_despite_old_search_path(self):
        source = r'''
import json, pathlib, runpy, sys
sys.dont_write_bytecode = True
framework, poison, metadata_probe = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(poison))
sys.path.append(str(framework))
def no_gpu_import(event, args):
    if event == "import" and (args[0] == "cupy" or args[0].startswith("cupy.")):
        raise AssertionError("layout import must not load a GPU backend")
sys.addaudithook(no_gpu_import)
loaded = {}
example_paths = list((framework / "harness_engineering/examples").glob("*_candidate.py"))
# Include a temporary binary AppleDouble fixture without changing project files.
example_paths.append(metadata_probe)
for path in sorted(path for path in example_paths if not path.name.startswith("._")):
    module = runpy.run_path(str(path), run_name="layout_import_probe")
    assert module["ROOT"] == framework
    assert module["HF_ROOT"] == framework.parent
    loaded[path.name] = module
import hf, gpu_hf, make_alkane_xyz, harness_engineering
assert loaded["rhf_direct_candidate.py"]["Hartree_Fock"] is hf.Hartree_Fock
assert loaded["gpu_rhf_candidate.py"]["GPUFriendlyRHF"] is gpu_hf.GPUFriendlyRHF
assert loaded["linear_get_jk_candidate.py"]["alkane_coordinates"] is make_alkane_xyz.alkane_coordinates
assert len(make_alkane_xyz.alkane_coordinates(2)) == 8
print(json.dumps({"examples": sorted(loaded), "sources": {
    name: str(pathlib.Path(module.__file__).resolve())
    for name, module in (("hf", hf), ("gpu_hf", gpu_hf),
                         ("make_alkane_xyz", make_alkane_xyz),
                         ("harness_engineering", harness_engineering))}}))
'''
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            poison = temporary / "old-checkout"
            (poison / "harness_engineering").mkdir(parents=True)
            for path in (poison / "harness_engineering/__init__.py",
                         poison / "hf.py", poison / "gpu_hf.py", poison / "make_alkane_xyz.py"):
                path.write_text('raise RuntimeError("old checkout must not be imported")\n')
            metadata_probe = temporary / "._foo_candidate.py"
            metadata_probe.write_bytes(b"\x00AppleDouble metadata\x00")
            result = self.probe(source, FRAMEWORK_ROOT, poison, metadata_probe, cwd=temporary)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["examples"]), 10)
        for name in ("hf", "gpu_hf", "make_alkane_xyz"):
            self.assertEqual(Path(report["sources"][name]), HF_ROOT / f"{name}.py")
        self.assertEqual(Path(report["sources"]["harness_engineering"]),
                         FRAMEWORK_ROOT / "harness_engineering/__init__.py")

    def test_direct_example_rejects_an_already_cached_old_framework(self):
        source = r'''
import pathlib, runpy, sys, types
sys.dont_write_bytecode = True
framework = pathlib.Path(sys.argv[1])
old = types.ModuleType("harness_engineering")
old.__file__ = str(pathlib.Path.cwd() / "old/harness_engineering/__init__.py")
old.__path__ = [str(pathlib.Path(old.__file__).parent)]
sys.modules["harness_engineering"] = old
runpy.run_path(str(framework / "harness_engineering/examples/rhf_direct_candidate.py"),
               run_name="layout_import_probe")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.probe(source, FRAMEWORK_ROOT, cwd=directory)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ImportError: harness_engineering is already loaded outside this harness framework",
                      result.stderr)

    def test_cached_old_submodules_and_parent_programs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("harness_engineering.training_hf", "hf", "gpu_hf", "make_alkane_xyz"):
                old = ModuleType(name)
                old.__file__ = str(Path(directory) / (name.replace(".", "/") + ".py"))
                with self.subTest(module=name), patch.dict(sys.modules, {name: old}), \
                        self.assertRaises(ImportError):
                    paths.configure_import_paths()
        package = sys.modules["harness_engineering"]
        with patch.object(package, "__path__", [str(FRAMEWORK_ROOT / "old")]), \
                self.assertRaises(ImportError):
            paths.configure_import_paths()

    def test_both_loop_tasks_protect_the_real_parent_hf(self):
        for filename in ("occupy_density_oracle.json", "occupy_density_budget.json"):
            task = json.loads((FRAMEWORK_ROOT / "harness/tasks" / filename).read_text())
            forbidden = task["generation"]["forbidden"]
            with self.subTest(task=filename):
                self.assertIn("../hf.py", forbidden)
                self.assertNotIn("hf.py", forbidden)
                before = snapshot_files(FRAMEWORK_ROOT, forbidden)
                self.assertEqual(before["../hf.py"], hashlib.sha256((HF_ROOT / "hf.py").read_bytes()).hexdigest())
                self.assertEqual(len(before), len(forbidden))
        # Exercise change detection without changing the production solver.
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            framework = outer / "harness_framework"
            framework.mkdir()
            solver = outer / "hf.py"
            solver.write_text("original solver\n")
            before = snapshot_files(framework, ["../hf.py"])
            solver.write_text("changed solver\n")
            self.assertEqual(changed_files(before, snapshot_files(framework, ["../hf.py"])), ["../hf.py"])

    def test_generation_defaults_resolve_framework_resources_from_another_cwd(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(generation_entry, "Campaign") as campaign, \
                    patch.object(generation_entry, "load_json", return_value={}), \
                    redirect_stdout(io.StringIO()):
                os.chdir(directory)
                campaign.return_value.run.return_value = {"status": "waiting_for_environment"}
                code = generation_entry.main(["run", "--output", "new-campaign",
                                              "--executor-config", "mock-executor.json"])
                self.assertEqual(code, 2)
                spec, holdout = campaign.return_value.run.call_args.args
                self.assertEqual(spec, FRAMEWORK_ROOT / "harness/generation/rhf.json")
                self.assertEqual(holdout, FRAMEWORK_ROOT / "harness/generation/holdout_rhf_v1.json")
        finally:
            os.chdir(original)

    def test_trusted_cpu_reference_subprocess_finds_framework_package_from_another_cwd(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                os.chdir(root)
                prepared = prepare_case({"id": "layout-he-exact", "atom": "He 0 0 0", "basis": "sto-3g"},
                                        "exact", root / "input", root / "reference", memory_limit_mb=128)
                execution = prepared["reference_execution"]
                self.assertEqual(execution["returncode"], 0)
                self.assertTrue(execution["independent_process"])
                self.assertTrue(Path(prepared["problem_path"]).is_file())
                self.assertTrue(Path(prepared["reference_path"]).is_file())
                self.assertEqual(prepared["identity"]["nelectron"], 2)
                self.assertEqual(prepared["identity"]["nao"], 1)
        finally:
            os.chdir(original)


if __name__ == "__main__":
    unittest.main()
