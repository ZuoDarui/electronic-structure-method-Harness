"""Single-shot accuracy gate: one program per run, small molecules, PySCF references.

Uses the real PySCF reference worker, the real worker process and the real
verifier on two tiny molecules. The model is replaced by an explicit double.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from harness_engineering.generation import single
from harness_engineering.generation.generator import GenerationError
from harness_engineering.generation.storage import load_json


ROOT = Path(__file__).resolve().parents[1]
GOOD = (ROOT / "tests/fixtures/rhf_numpy_solution.py").read_text()
TASK = load_json(ROOT / "harness/generation/rhf_small_accuracy.json")
EXECUTION = {"cpu_executor": "subprocess", "allow_unisolated_host_execution": True,
             "python": sys.executable, "timeout_s": 120}


def small_task(directory, ids=("h2-sto3g", "water-631gs")):
    spec = copy.deepcopy(TASK)
    spec["cases"] = [case for case in spec["cases"] if case["id"] in ids]
    path = Path(directory) / "task.json"
    path.write_text(json.dumps(spec))
    return path


class Generator:
    calls = []

    def __init__(self, source=None, error=None):
        self.source, self.error = source, error

    def factory(self, config):
        owner = self

        class Bound:
            def generate(self, context, output_dir):
                Generator.calls.append(copy.deepcopy(context))
                if owner.error:
                    raise GenerationError(*owner.error)
                return {"files": [{"path": "solution.py", "content": owner.source}],
                        "rationale": "test", "expected_effect": "test"}
        return Bound()


class SingleShotTests(unittest.TestCase):
    def setUp(self):
        Generator.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.task = small_task(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, source=None, error=None, name="run", **kwargs):
        generator = Generator(source, error)
        runner = single.SingleShotRun(self.dir / name, generator_factory=generator.factory)
        return runner.run(self.task, execution_config=EXECUTION,
                          generation_config={"model": "test-model"}, **kwargs)

    def test_correct_program_accepted_with_exactly_one_generation(self):
        result = self.run_with(GOOD)
        self.assertEqual(result["status"], "accepted", result["reasons"])
        self.assertEqual(len(Generator.calls), 1)
        self.assertEqual(result["case_count"], 4)  # 2 molecules x exact/df
        self.assertTrue((self.dir / "run/accepted/solution.py").is_file())
        self.assertLess(result["max_metrics"]["energy_max_abs"], 1e-8)
        report = (self.dir / "run/report.md").read_text()
        self.assertIn("h2-sto3g-exact", report)

    def test_model_context_has_no_references_gpu_or_speed_gate(self):
        self.run_with(GOOD)
        context = json.dumps(Generator.calls[0])
        self.assertNotIn("reference.npz", context)
        self.assertNotIn("private", context)
        self.assertIn("backend is always the string 'numpy'", Generator.calls[0]["contract"])
        self.assertNotIn("dev_feedback", Generator.calls[0])

    def test_wrong_exchange_is_rejected_not_repaired(self):
        wrong = GOOD.replace("return problem[\"hcore\"] + j - 0.5 * k", "return problem[\"hcore\"] + j - 0.45 * k")
        self.assertNotEqual(wrong, GOOD)
        result = self.run_with(wrong)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(len(Generator.calls), 1)
        self.assertTrue(any("energy_max_abs" in reason for reason in result["reasons"]))
        self.assertFalse((self.dir / "run/accepted").exists())

    def test_wrong_jk_definition_is_rejected(self):
        wrong = GOOD.replace('"prqs,rs->pq"', '"pqrs,rs->pq"')
        result = self.run_with(wrong)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(any("jk_max_abs" in reason for reason in result["reasons"]))

    def test_crashing_program_is_a_scientific_rejection(self):
        crash = GOOD.replace("def solve(problem, options, backend):",
                             "def solve(problem, options, backend):\n    raise ValueError('boom')\n\ndef _unused(problem, options, backend):")
        result = self.run_with(crash)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(all(case["status"] == "failed" for case in result["cases"]))

    def test_fake_convergence_flag_rejected(self):
        lazy = GOOD.replace('max_cycle = int(options.get("max_cycle", 100))', "max_cycle = 1")
        result = self.run_with(lazy)
        self.assertEqual(result["status"], "rejected")

    def test_forbidden_import_is_generation_failure(self):
        result = self.run_with("import os\n" + GOOD)
        self.assertEqual(result["status"], "generation_failed")
        self.assertFalse((self.dir / "run/cases").exists())

    def test_api_failure_does_not_retry_with_a_second_program(self):
        result = self.run_with(error=("network_error", "test outage"))
        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(len(Generator.calls), 1)

    def test_previous_run_feedback_is_public_only(self):
        wrong = GOOD.replace("- 0.5 * k", "- 0.45 * k")
        self.run_with(wrong, name="first")
        Generator.calls = []
        self.run_with(GOOD, name="second", previous_run=self.dir / "first")
        context = Generator.calls[0]
        self.assertEqual(context["dev_feedback"]["status"], "rejected")
        self.assertIn("0.45", context["current_source"])
        self.assertNotIn("reference", json.dumps(context["dev_feedback"]).replace("reference_", ""))

    def test_supplied_candidate_skips_model(self):
        path = self.dir / "solution.py"
        path.write_text(GOOD)
        result = single.SingleShotRun(self.dir / "supplied").run(
            self.task, execution_config=EXECUTION, candidate_source=path)
        self.assertEqual(result["status"], "accepted", result["reasons"])
        self.assertEqual(result["mode"], "supplied_candidate")

    def test_output_directory_must_be_new(self):
        (self.dir / "used").mkdir()
        (self.dir / "used/x").write_text("x")
        with self.assertRaises(ValueError):
            single.SingleShotRun(self.dir / "used").run(self.task, execution_config=EXECUTION,
                                                        candidate_source=self.dir / "x")


class SpecLimitTests(unittest.TestCase):
    def test_default_task_is_small_and_numpy_only(self):
        spec = single.validate_accuracy_spec(TASK)
        self.assertEqual(spec["accuracy_only"]["backend"], "numpy")
        self.assertGreaterEqual(len(spec["cases"]), 10)
        self.assertEqual(spec["generation"]["max_rounds"], 1)

    def test_large_molecule_rejected_before_any_computation(self):
        spec = copy.deepcopy(TASK)
        spec["cases"] = [{"id": "big", "atom": "; ".join(f"H 0 0 {i}" for i in range(20)), "basis": "sto-3g"}]
        with self.assertRaisesRegex(ValueError, "small-molecule limit"):
            single.validate_accuracy_spec(spec)

    def test_nao_limit_enforced_after_reference_preparation(self):
        spec = copy.deepcopy(TASK)
        spec["accuracy_only"]["max_nao"] = 5
        spec["cases"] = [case for case in spec["cases"] if case["id"] == "water-631gs"]
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.json"
            task.write_text(json.dumps(spec))
            source = Path(directory) / "solution.py"
            source.write_text(GOOD)
            result = single.SingleShotRun(Path(directory) / "run").run(
                task, execution_config=EXECUTION, candidate_source=source)
        self.assertEqual(result["status"], "invalid_task")
        self.assertTrue(any("exceeds the small-molecule limit" in r for r in result["reasons"]))

    def test_gpu_backend_not_allowed(self):
        spec = copy.deepcopy(TASK)
        spec["accuracy_only"]["backend"] = "cupy"
        with self.assertRaises(ValueError):
            single.validate_accuracy_spec(spec)

    def test_subprocess_executor_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            single.make_executor({"cpu_executor": "subprocess"})


if __name__ == "__main__":
    unittest.main()
