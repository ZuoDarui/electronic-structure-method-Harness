"""Controller protocol tests with explicit simulated model/runtime dependencies.

These tests prove state and evidence behavior, not generated scientific success.
Real numerical verifier and worker checks live in separate test modules.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from harness_engineering.generation.campaign import Campaign, STAGES
from harness_engineering.generation.generator import GenerationError
from harness_engineering.generation.storage import atomic_json, digest, load_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''import numpy as np

def build_jk(problem, density, backend):
    raise NotImplementedError("simulated controller test")

def solve(problem, options, backend):
    raise NotImplementedError("simulated controller test")
'''


class ScienceDouble:
    def __init__(self):
        self.preparations = []

    def prepare_case(self, case, mode, input_dir, reference_dir, **kwargs):
        self.preparations.append(case["id"])
        input_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        problem, reference = input_dir / (case["id"] + ".npz"), reference_dir / (case["id"] + ".npz")
        problem.write_text("fake public input " + case["id"])
        reference.write_text("private answer must not reach generator")
        return {"id": case["id"], "problem_path": str(problem), "input_sha256": digest(problem),
                "reference_path": str(reference), "reference_sha256": digest(reference),
                "identity": {"physical_input_sha256": hashlib.sha256(json.dumps(
                    {"atom": case["atom"], "basis": case["basis"], "mode": mode}, sort_keys=True).encode()).hexdigest()}}

    def verify_case(self, prepared, path, tolerances, execution):
        good = execution["status"] == "passed"
        return {"case_id": prepared["id"], "status": "passed" if good else "failed",
                "metrics": {"error": 0 if good else 1}, "reasons": [] if good else ["test numerical fault"]}

    def verify_suite(self, expected, records):
        passed = bool(expected) and len(expected) == len(records) and all(r["status"] == "passed" for r in records)
        return {"status": "passed" if passed else "failed", "passed": passed,
                "metrics": {"error": max((r["metrics"]["error"] for r in records), default=0)},
                "reasons": [] if passed else ["test suite rejected"]}

    def make_challenge_cases(self, spec):
        base = spec["cases"][0]
        return [{**base, "id": base["id"] + "-rigid-" + mode, "mode": mode,
                 "challenge": {"kind": "rigid_motion", "base_case_id": base["id"] + "-" + mode}}
                for mode in spec["integral_modes"]]

    def verify_rigid_motion(self, first, second, tolerance):
        passed = first["status"] == second["status"] == "passed"
        return {"status": "passed" if passed else "failed", "passed": passed,
                "metrics": {}, "reasons": [] if passed else ["test rigid transform failed"]}

    def verify_nonconverged_case(self, prepared, path, tolerances, execution, max_cycle):
        return self.verify_case(prepared, path, tolerances, execution)


class RuntimeDouble:
    def __init__(self):
        self.available = True
        self.failure_stage = None
        self.fail_once = False
        self.interrupt_once = False
        self.interrupt_after_collection = False
        self.orphan_once = False
        self.calls = []
        self.generation_calls = []
        self.identity = "fixed-test-image"
        self.leak = False
        self.raise_generation = None
        self.lose_first_generation = False

    def executor(self, config):
        owner = self
        class Executor:
            def preflight(self, backend):
                return {"status": "passed" if owner.available else "invalid_runtime",
                        "image": owner.identity, "probe": {"backend": backend}, "reasons": ["test daemon unavailable"]}

            def run(self, candidate_dir, problem_path, output_dir, backend, options):
                owner.calls.append(str(output_dir))
                stage = output_dir.parent.parent.name
                if owner.interrupt_once:
                    owner.interrupt_once = False
                    raise KeyboardInterrupt()
                output_dir.mkdir(parents=True, exist_ok=True)
                if owner.orphan_once:
                    owner.orphan_once = False
                    (output_dir / "partial.txt").write_text("interrupted evidence")
                    raise KeyboardInterrupt()
                path = output_dir / "result.npz"
                path.write_text("test raw arrays")
                failed = stage == owner.failure_stage
                if failed and owner.fail_once:
                    owner.failure_stage = None
                artifacts = {"result.npz": digest(path)}
                if backend == "cupy" and config.get("require_external_trace"):
                    for name in ("cuda_trace.sqlite", "cuda_trace.nsys-rep"):
                        (output_dir / name).write_text("simulated test trace")
                        artifacts[name] = digest(output_dir / name)
                receipt = {"status": "failed" if failed else "passed", "returncode": 1 if failed else 0,
                        "output_path": str(path), "candidate_sha256": digest(candidate_dir / "solution.py"),
                        "input_sha256": digest(problem_path), "runtime": self.preflight(backend),
                        "hashes": {"artifacts": artifacts}, "trace": {"external": {"valid": True}}}
                if owner.interrupt_after_collection:
                    owner.interrupt_after_collection = False
                    atomic_json(output_dir / "execution.json", receipt)
                    raise KeyboardInterrupt()
                return receipt
        return Executor()

    def generator(self, config):
        owner = self
        class Generator:
            def generate(self, context, output_dir):
                owner.generation_calls.append(copy.deepcopy(context))
                serialized = json.dumps(context)
                if "private answer" in serialized or "private-holdout" in serialized:
                    owner.leak = True
                if owner.lose_first_generation:
                    owner.lose_first_generation = False
                    raise KeyboardInterrupt()
                if owner.raise_generation:
                    raise GenerationError(owner.raise_generation, "test API fault")
                output_dir.mkdir(parents=True, exist_ok=True)
                attempt = output_dir / "attempt-01"
                attempt.mkdir()
                atomic_json(attempt / "response.json", {"usage": {"input_tokens": 10, "output_tokens": 20}})
                return {"files": [{"path": "solution.py", "content": SOURCE + f"\n# round {context['round']}\n"}],
                        "rationale": "controller protocol test fixture", "expected_effect": "simulated runtime verdict"}
        return Generator()


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "campaign"
        self.runtime = RuntimeDouble()
        self.science = ScienceDouble()
        public = load_json(ROOT / "harness/generation/rhf.json")
        public["cases"] = public["cases"][:1]
        public["generation"]["max_rounds"] = 2
        private = copy.deepcopy(public)
        private["name"] = "private-holdout"
        private["cases"][0]["id"] = "private-holdout"
        private["cases"][0]["atom"] = "H 0 0 0; H 0 0 0.843"
        self.spec, self.holdout = self.directory / "task.json", self.directory / "holdout.json"
        atomic_json(self.spec, public)
        atomic_json(self.holdout, private)

    def tearDown(self):
        self.temporary.cleanup()

    def campaign(self):
        return Campaign(self.root, generator_factory=self.runtime.generator,
                        executor_factory=self.runtime.executor, science_module=self.science)

    def run_campaign(self):
        return self.campaign().run(self.spec, self.holdout, model="gpt-6.1-sol", reasoning_effort="ultra")

    def test_empty_start_freeze_four_gates_promote_and_recompute(self):
        result = self.run_campaign()
        self.assertEqual(result["status"], "accepted")
        self.assertTrue((self.root / "accepted/solution.py").is_file())
        self.assertIn("NotImplementedError", (self.root / "initial/solution.py").read_text())
        self.assertEqual(set(result["stage_reports"]), set(STAGES))
        self.assertEqual(len(self.runtime.calls), 13)
        self.assertEqual(result["api_usage"], {"input_tokens": 10, "output_tokens": 20})
        self.assertTrue(self.campaign().evaluate()["accepted"])
        self.assertFalse(self.runtime.leak)
        self.assertEqual(len(self.runtime.generation_calls), 1)

    def test_numerical_repair_has_new_snapshot_and_actual_rollback(self):
        self.runtime.failure_stage, self.runtime.fail_once = "dev_cpu", True
        result = self.run_campaign()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["round"], 2)
        self.assertNotEqual(digest(self.root / "rounds/0001/candidate/solution.py"),
                            digest(self.root / "rounds/0002/candidate/solution.py"))
        self.assertIn('"event": "rolled_back"', (self.root / "events.jsonl").read_text())
        self.assertEqual(self.runtime.generation_calls[1]["dev_feedback"]["status"], "failed")

    def test_final_failure_never_feeds_hidden_results_to_generator(self):
        self.runtime.failure_stage = "holdout_cpu"
        result = self.run_campaign()
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(len(self.runtime.generation_calls), 1)
        self.assertFalse(self.runtime.leak)
        self.assertEqual(self.campaign().resume()["status"], "rejected")
        self.assertEqual(len(self.runtime.generation_calls), 1)

    def test_missing_container_is_resumable_and_does_not_spend_api_budget(self):
        self.runtime.available = False
        result = self.run_campaign()
        self.assertEqual(result["status"], "waiting_for_environment")
        self.assertEqual(self.runtime.generation_calls, [])
        self.assertEqual(self.science.preparations, [])
        self.runtime.available = True
        self.assertEqual(self.campaign().resume()["status"], "accepted")

    def test_explicit_deferred_gpu_mode_completes_cpu_then_waits_without_promoting(self):
        owner = self.runtime
        original = owner.executor
        gpu_ready = [False]
        def factory(config):
            wrapped = original(config)
            class Executor:
                def preflight(self, backend):
                    if backend == "cupy" and not gpu_ready[0]:
                        return {"status": "invalid_runtime", "reasons": ["GPU runtime unavailable"]}
                    return wrapped.preflight(backend)
                def run(self, *args, **kwargs):
                    return wrapped.run(*args, **kwargs)
            return Executor()
        def campaign():
            return Campaign(self.root, generator_factory=owner.generator, executor_factory=factory,
                            science_module=self.science)
        state = campaign().run(self.spec, self.holdout, model="gpt-6.1-sol", reasoning_effort="max",
                               execution_config={"defer_gpu_preflight": True})
        self.assertEqual(state["status"], "waiting_for_environment")
        self.assertEqual(state["phase"], "dev_gpu")
        self.assertEqual(set(state["stage_reports"]), {"dev_cpu", "dev_properties"})
        self.assertFalse(state["holdout_started"])
        self.assertFalse((self.root / "accepted").exists())
        self.assertEqual(len(owner.generation_calls), 1)
        self.assertEqual(set(owner.generation_calls[0]["public_prompt"]["target_runtime"]), {"numpy"})
        gpu_ready[0] = True
        self.assertEqual(campaign().resume()["status"], "accepted")
        self.assertEqual(len(owner.generation_calls), 1)

    def test_deferred_gpu_mode_still_requires_a_qualified_cpu_container(self):
        self.runtime.available = False
        state = self.campaign().run(self.spec, self.holdout, model="gpt-6.1-sol",
                                     execution_config={"defer_gpu_preflight": True})
        self.assertEqual(state["status"], "waiting_for_environment")
        self.assertEqual(state["round"], 0)
        self.assertEqual(self.runtime.generation_calls, [])

    def test_explicit_user_cpu_only_boundary_makes_zero_gpu_calls_until_enabled(self):
        original = self.runtime.executor
        calls = []
        def factory(config):
            wrapped = original(config)
            class Executor:
                def preflight(self, backend):
                    calls.append(("preflight", backend))
                    return wrapped.preflight(backend)
                def run(self, candidate_dir, problem_path, output_dir, backend, options):
                    calls.append(("run", backend))
                    return wrapped.run(candidate_dir, problem_path, output_dir, backend, options)
            return Executor()
        def campaign():
            return Campaign(self.root, generator_factory=self.runtime.generator,
                            executor_factory=factory, science_module=self.science)
        state = campaign().run(self.spec, self.holdout, model="gpt-6.1-sol",
                               execution_config={"stop_before_gpu": True})
        self.assertEqual(state["status"], "waiting_for_gpu_authorization")
        self.assertEqual(state["phase"], "dev_gpu")
        self.assertFalse(any(backend == "cupy" for _, backend in calls))
        self.assertFalse(state["holdout_started"])
        self.assertEqual(campaign().resume()["status"], "waiting_for_gpu_authorization")
        self.assertFalse(any(backend == "cupy" for _, backend in calls))
        self.assertEqual(campaign().resume(enable_gpu=True)["status"], "accepted")
        self.assertTrue(any(backend == "cupy" for _, backend in calls))
        self.assertEqual(len(self.runtime.generation_calls), 1)

    def test_extended_budget_is_frozen_and_cpu_only_resume_does_not_expand_authorization(self):
        original_executor, original_generator = self.runtime.executor, self.runtime.generator
        preflights, generation_configs = [], []
        def executor_factory(config):
            wrapped = original_executor(config)
            class Executor:
                def preflight(self, backend):
                    preflights.append(backend)
                    return wrapped.preflight(backend)
                def run(self, *args, **kwargs):
                    return wrapped.run(*args, **kwargs)
            return Executor()
        def generator_factory(config):
            generation_configs.append(copy.deepcopy(config))
            return original_generator(config)
        def campaign():
            return Campaign(self.root, generator_factory=generator_factory,
                            executor_factory=executor_factory, science_module=self.science)
        extended = {"budget_profile": "extended", "max_output_tokens": 32768,
                    "timeout_s": 1200, "max_retries": 0, "streaming": True}
        state = campaign().run(self.spec, self.holdout, model="gpt-6.1-sol", reasoning_effort="max",
                               execution_config={"stop_before_gpu": True}, generation_config=extended)
        self.assertEqual(state["status"], "waiting_for_gpu_authorization")
        config_path = self.root / "generation_config.json"
        frozen_config = load_json(config_path)
        for key, value in extended.items():
            self.assertEqual(frozen_config[key], value)
            self.assertEqual(generation_configs[0][key], value)
        self.assertEqual(state["trusted_identity"][str(config_path.resolve())], digest(config_path))
        config_hash = digest(config_path)
        resumed = campaign().resume()
        self.assertEqual(resumed["status"], "waiting_for_gpu_authorization")
        self.assertEqual(resumed["phase"], "dev_gpu")
        self.assertFalse(resumed["holdout_started"])
        self.assertFalse(resumed.get("gpu_authorized", False))
        self.assertEqual(set(preflights), {"numpy"})
        self.assertEqual(len(generation_configs), 1)
        self.assertEqual(digest(config_path), config_hash)
        # Larger numbers without explicit extended policy fail before even a
        # mocked runtime probe, integral preparation or generation can start.
        for index, illegal in enumerate(({"max_output_tokens": 32768},
                                          {"budget_profile": "standard", "timeout_s": 1200})):
            self.root = self.directory / f"illegal-standard-{index}"
            before = (len(preflights), len(self.science.preparations), len(generation_configs))
            with self.assertRaises(GenerationError):
                campaign().run(self.spec, self.holdout, model="gpt-6.1-sol", generation_config=illegal)
            self.assertEqual(before, (len(preflights), len(self.science.preparations), len(generation_configs)))
            self.assertFalse((self.root / "state.json").exists())

    def test_interrupted_candidate_stage_resumes_without_second_model_call(self):
        self.runtime.interrupt_once = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        result = self.campaign().resume()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(self.runtime.generation_calls), 1)
        self.assertEqual(len(self.science.preparations), 6)

    def test_receipt_collected_before_interruption_is_recovered_without_rerun(self):
        self.runtime.interrupt_after_collection = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        self.assertEqual(self.campaign().resume()["status"], "accepted")
        self.assertEqual(len(self.runtime.calls), 13)
        self.assertIn("recovered_execution_receipt", (self.root / "events.jsonl").read_text())

    def test_orphan_output_is_preserved_then_fresh_execution_resumes(self):
        self.runtime.orphan_once = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        self.assertEqual(self.campaign().resume()["status"], "accepted")
        orphan = list((self.root / "rounds").glob("*/dev_cpu/*/interrupted-output-*/partial.txt"))
        self.assertEqual(len(orphan), 1)
        self.assertEqual(orphan[0].read_text(), "interrupted evidence")
        self.assertEqual(len(self.runtime.calls), 14)

    def test_lost_api_request_consumes_round_without_silent_replay(self):
        self.runtime.lose_first_generation = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        result = self.campaign().resume()
        self.assertEqual(result["round"], 2)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(self.runtime.generation_calls), 2)

    def test_budget_exhaustion_keeps_failed_evidence(self):
        self.runtime.failure_stage = "dev_cpu"
        result = self.run_campaign()
        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["round"], 2)
        self.assertEqual(len(list((self.root / "rounds").glob("*/dev_cpu/suite.json"))), 2)
        self.assertFalse((self.root / "accepted").exists())

    def test_configuration_or_api_error_waits_instead_of_burning_all_rounds(self):
        for status in ("config_error", "http_error", "network_error", "output_budget_exceeded", "invalid_usage",
                       "invalid_stream", "stream_limit"):
            with self.subTest(status=status):
                self.root = self.directory / status
                self.runtime.raise_generation = status
                result = self.run_campaign()
                self.assertEqual(result["status"], "waiting_for_environment")
                self.assertEqual(result["round"], 1)

    def test_frozen_task_or_reference_change_prevents_resume(self):
        self.runtime.available = False
        self.run_campaign()
        with (self.root / "spec.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()

    def test_changed_runtime_cannot_reuse_old_results(self):
        self.run_campaign()
        state = load_json(self.root / "state.json")
        state.update(status="waiting_for_environment", phase="dev_cpu")
        atomic_json(self.root / "state.json", state)
        self.runtime.identity = "changed-image"
        result = self.campaign().resume()
        self.assertEqual(result["status"], "waiting_for_environment")
        self.assertIn("identity", result["blocked_reason"])

    def test_tampered_artifacts_fail_offline_reassessment(self):
        self.run_campaign()
        path = next((self.root / "rounds").glob("*/holdout_gpu/*/output/result.npz"))
        path.write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().evaluate()

    def test_blank_model_and_nonempty_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "model"):
            self.campaign().run(self.spec, self.holdout, model="")
        self.root.mkdir(exist_ok=True)
        (self.root / "unrelated.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.run_campaign()

    def test_same_physical_holdout_cannot_be_reused_in_a_second_campaign(self):
        self.assertEqual(self.run_campaign()["status"], "accepted")
        self.root = self.directory / "second-campaign"
        state = self.run_campaign()
        self.assertEqual(state["status"], "rejected")
        self.assertFalse(state["holdout_started"])
        self.assertIn("holdout_reuse_rejected", (self.root / "events.jsonl").read_text())

    def test_renaming_a_development_input_does_not_make_a_blind_case(self):
        holdout = load_json(self.holdout)
        holdout["cases"][0]["atom"] = load_json(self.spec)["cases"][0]["atom"]
        atomic_json(self.holdout, holdout)
        with self.assertRaisesRegex(ValueError, "duplicates a resolved physical"):
            self.run_campaign()
        self.assertEqual(len(self.runtime.generation_calls), 0)


if __name__ == "__main__":
    unittest.main()
