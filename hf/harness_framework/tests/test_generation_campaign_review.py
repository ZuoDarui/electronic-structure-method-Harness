"""Independent state-machine adversarial checks using explicit runtime doubles.

These checks test control/evidence behavior only. They do not qualify generated
HF code, containers, GPU execution, or any model response scientifically.
"""
import copy
from pathlib import Path
import tempfile
import unittest

from harness_engineering.generation.campaign import Campaign
from harness_engineering.generation.storage import atomic_json, load_json
from test_generation_campaign import RuntimeDouble, ScienceDouble


ROOT = Path(__file__).resolve().parents[1]


class AuditedScienceDouble(ScienceDouble):
    def __init__(self):
        super().__init__()
        self.rigid_calls = 0
        self.reject_rigid = False

    def verify_rigid_motion(self, first, second, tolerance):
        self.rigid_calls += 1
        if self.reject_rigid:
            return {"status": "failed", "passed": False,
                    "metrics": {"candidate_energy_invariance_abs": 0.1},
                    "reasons": ["independent review injected broken rigid invariance"]}
        return super().verify_rigid_motion(first, second, tolerance)


class AuditedRuntimeDouble(RuntimeDouble):
    def __init__(self):
        super().__init__()
        self.interrupt_at_stage = None
        self.remove_final_trace_receipt = False
        self.invalidate_final_trace = False
        self.unavailable_receipt_once = False

    def executor(self, config):
        wrapped = super().executor(config)
        owner = self

        class Executor:
            def preflight(self, backend):
                return wrapped.preflight(backend)

            def run(self, candidate_dir, problem_path, output_dir, backend, options):
                receipt = wrapped.run(candidate_dir, problem_path, output_dir, backend, options)
                stage = output_dir.parent.parent.name
                if stage == "holdout_gpu":
                    if owner.remove_final_trace_receipt:
                        receipt["hashes"]["artifacts"].pop("cuda_trace.sqlite")
                    if owner.invalidate_final_trace:
                        receipt["trace"]["external"]["valid"] = False
                    if owner.unavailable_receipt_once:
                        owner.unavailable_receipt_once = False
                        receipt.update(status="invalid_runtime", reason="temporary profiler collection unavailable")
                        atomic_json(output_dir / "execution.json", receipt)
                if stage == owner.interrupt_at_stage:
                    owner.interrupt_at_stage = None
                    atomic_json(output_dir / "execution.json", receipt)
                    raise KeyboardInterrupt("interrupted after trusted executor collection")
                return receipt

        return Executor()


class CampaignReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "campaign"
        self.runtime = AuditedRuntimeDouble()
        self.science = AuditedScienceDouble()
        dev = load_json(ROOT / "harness/generation/rhf.json")
        dev["cases"] = dev["cases"][:1]
        dev["generation"]["max_rounds"] = 2
        holdout = copy.deepcopy(dev)
        holdout["name"] = "review-holdout"
        holdout["cases"][0].update(id="review-held-case", atom="H 0 0 0; H 0 0 0.843")
        self.spec, self.holdout = self.directory / "task.json", self.directory / "holdout.json"
        atomic_json(self.spec, dev)
        atomic_json(self.holdout, holdout)

    def tearDown(self):
        self.temporary.cleanup()

    def campaign(self):
        return Campaign(self.root, generator_factory=self.runtime.generator,
                        executor_factory=self.runtime.executor, science_module=self.science)

    def run_campaign(self):
        return self.campaign().run(self.spec, self.holdout, model="gpt-6.1-sol", reasoning_effort="max")

    def test_completed_stage_tampering_is_rejected_before_resuming_any_execution(self):
        self.run_campaign()
        state = load_json(self.root / "state.json")
        path = Path(state["stage_reports"]["dev_cpu"]["path"])
        path.write_text(path.read_text() + " ")
        calls_before = len(self.runtime.calls)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()
        self.assertEqual(len(self.runtime.calls), calls_before)

    def test_old_rejected_snapshot_remains_protected_after_later_candidate_passes(self):
        self.runtime.failure_stage, self.runtime.fail_once = "dev_cpu", True
        result = self.run_campaign()
        self.assertEqual(result["round"], 2)
        old = self.root / "rounds/0001/candidate/solution.py"
        old.write_text(old.read_text() + "\n# changed old evidence\n")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()

    def test_missing_trace_from_interrupted_collection_cannot_be_reused(self):
        self.runtime.interrupt_at_stage = "holdout_gpu"
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        trace = next((self.root / "rounds").glob("*/holdout_gpu/*/output/cuda_trace.sqlite"))
        trace.unlink()
        calls_before = len(self.runtime.calls)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()
        self.assertEqual(len(self.runtime.calls), calls_before)
        self.assertFalse((self.root / "accepted").exists())

    def test_corrupted_raw_output_from_interrupted_collection_cannot_be_reused(self):
        self.runtime.interrupt_at_stage = "holdout_gpu"
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign()
        path = next((self.root / "rounds").glob("*/holdout_gpu/*/output/result.npz"))
        path.write_text("altered after collection")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()
        self.assertFalse((self.root / "accepted").exists())

    def test_final_trace_boolean_without_artifact_receipt_does_not_pass(self):
        self.runtime.remove_final_trace_receipt = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "waiting_for_environment")
        self.assertIn("profiling artifacts", result["blocked_reason"])
        self.assertFalse((self.root / "accepted").exists())

    def test_final_trace_files_without_successful_external_validation_do_not_pass(self):
        self.runtime.invalidate_final_trace = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "waiting_for_environment")
        self.assertIn("valid external CUDA trace", result["blocked_reason"])
        self.assertFalse((self.root / "accepted").exists())

    def test_unavailable_durable_receipt_is_preserved_without_poisoning_future_resume(self):
        self.runtime.unavailable_receipt_once = True
        first = self.run_campaign()
        self.assertEqual(first["status"], "waiting_for_environment")
        calls_before = len(self.runtime.calls)
        resumed = self.campaign().resume()
        self.assertEqual(resumed["status"], "accepted")
        self.assertGreater(len(self.runtime.calls), calls_before)
        self.assertEqual(len(self.runtime.generation_calls), 1)
        # The initial failed collection remains inspectable somewhere under
        # the case directory after the successful run replaces its active slot.
        preserved = [path for path in (self.root / "rounds").rglob("execution.json")
                     if "temporary profiler collection unavailable" in path.read_text()]
        self.assertTrue(preserved)

    def test_failed_metamorphic_comparison_blocks_freeze_and_promote(self):
        self.science.reject_rigid = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "exhausted")
        self.assertFalse(result["holdout_started"])
        self.assertGreater(self.science.rigid_calls, 0)
        self.assertFalse((self.root / "frozen").exists())
        self.assertFalse((self.root / "accepted").exists())
        self.assertFalse(any("holdout_gpu" in name for name in self.runtime.calls))

    def test_offline_reassessment_replays_metamorphic_comparisons(self):
        self.run_campaign()
        previous = self.science.rigid_calls
        self.assertTrue(self.campaign().evaluate()["accepted"])
        self.assertEqual(self.science.rigid_calls, previous + 2)

    def test_final_promoted_source_is_bound_to_the_frozen_source(self):
        self.run_campaign()
        source = self.root / "accepted/solution.py"
        source.write_text(source.read_text() + "\n# altered promotion\n")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().resume()
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.campaign().evaluate()


if __name__ == "__main__":
    unittest.main()
