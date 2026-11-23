import json
from pathlib import Path
import sys

import numpy as np
import tempfile
import unittest

from harness_engineering.benchmark import benchmark, percentile
from harness_engineering.models import AcceptanceGate, CandidateRun, MethodSpec, MetricReport
from harness_engineering.cli import main as cli_main
from harness_engineering.runner import classify_failure, evaluate_candidate, run_candidate
from harness_engineering.examples.linear_get_jk_candidate import (
    estimate_linear_seconds,
    scaling_exponent,
    should_skip_linear,
)
from harness_engineering.training_hf import (
    TARGET_BASIS_FUNCTIONS,
    TARGET_SCF_ITERATIONS,
    TARGET_WALL_HOURS,
    density_asymmetry,
    density_electrons,
    occupy_density_reference,
    project_iteration_hours,
)
from harness_engineering.loop import (
    correct_despite_projection,
    decide_round,
    is_projection_reason,
    projections_stalled,
)
from harness_engineering.examples.water_cluster_guess_eigh_candidate import (
    estimate_eigh_seconds,
    generalized_residual,
    should_skip_eigh,
)
from harness_engineering.examples.water_cluster_jk_candidate import (
    estimate_jk_seconds,
    should_skip_jk,
)
from harness_engineering.water_clusters import (
    CLUSTER_COUNTS,
    cluster_atoms,
    cluster_shape,
    nao_mismatch_max,
)


ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_spec_round_trip(self):
        raw = {
            "name": "toy",
            "description": "test",
            "command": [sys.executable, "-c", "pass"],
            "cases": [{"name": "one"}],
            "correctness_tolerances": {"error": 1e-6},
            "required_metrics": ["error"],
        }
        spec = MethodSpec.from_dict(raw)
        self.assertEqual(spec.to_dict()["command"], raw["command"])
        self.assertEqual(spec.to_dict()["cases"], raw["cases"])

    def test_run_and_accept_report(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                "-c",
                "import json,os; json.dump({'metrics': {'error': 1e-8, 'all_finite': True}}, open(os.environ['HARNESS_REPORT_PATH'], 'w'))",
            ]
            spec = MethodSpec(
                name="toy",
                description="",
                command=tuple(command),
                correctness_tolerances={"error": 1e-6},
                required_metrics=("error",),
            )
            run = run_candidate(spec, directory)
            decision = evaluate_candidate(spec, run)
            self.assertTrue(decision.passed, decision.reasons)

    def test_report_dir_is_separate_from_candidate_workdir(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory) / "work"
            report_dir = Path(directory) / "reports"
            command = [
                sys.executable,
                "-c",
                "import json,os; json.dump({'metrics': {'error': 0.0}}, open(os.environ['HARNESS_REPORT_PATH'], 'w'))",
            ]
            spec = MethodSpec(name="separate", description="", command=tuple(command))
            run = run_candidate(spec, workdir, report_dir=report_dir)
            self.assertTrue(run.ok)
            self.assertEqual(Path(run.report_path).parent, report_dir.resolve())
            self.assertTrue(Path(run.report_path).exists())

    def test_cli_writes_acceptance_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            reports = root / "reports"
            task.write_text(json.dumps({
                "name": "cli-artifact",
                "description": "",
                "command": [
                    sys.executable, "-c",
                    "import json,os; json.dump({'metrics': {'error': 0.0}}, open(os.environ['HARNESS_REPORT_PATH'], 'w'))",
                ],
                "required_metrics": ["error"],
                "correctness_tolerances": {"error": 1e-6},
            }))
            rc = cli_main(["run", str(task), "--workdir", str(root), "--report-dir", str(reports)])
            self.assertEqual(rc, 0)
            acceptance = reports / "cli-artifact.acceptance.json"
            self.assertTrue(acceptance.exists())
            payload = json.loads(acceptance.read_text())
            self.assertTrue(payload["decision"]["passed"])

    def test_cli_appends_experiment_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            reports = root / "reports"

            def run_with(error: str) -> int:
                task.write_text(json.dumps({
                    "name": "cli-archive",
                    "description": "",
                    "command": [
                        sys.executable, "-c",
                        "import json,os; json.dump({'metrics': {'error': "
                        + error
                        + "}}, open(os.environ['HARNESS_REPORT_PATH'], 'w'))",
                    ],
                    "required_metrics": ["error"],
                    "correctness_tolerances": {"error": 1e-6},
                }))
                return cli_main([
                    "run", str(task), "--workdir", str(root), "--report-dir", str(reports),
                ])

            self.assertEqual(run_with("0.0"), 0)
            self.assertEqual(run_with("1e-9"), 0)
            records = [
                json.loads(line)
                for line in (reports / "experiments.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual([record["passed"] for record in records], [True, True])
            copies = list((reports / "runs").glob("*.acceptance.json"))
            self.assertEqual(len(copies), 2)
            archived_errors = [
                json.loads(copy.read_text())["decision"]["metrics"]["error"]
                for copy in copies
            ]
            self.assertCountEqual(archived_errors, [0.0, 1e-9])
            latest = json.loads((reports / "cli-archive.acceptance.json").read_text())
            self.assertEqual(latest["decision"]["metrics"]["error"], 1e-9)
            for record in records:
                self.assertEqual(record["task"], "cli-archive")
                self.assertEqual(record["status"], "completed")
                self.assertTrue(record["command"])
                self.assertTrue(Path(record["acceptance_path"]).is_file())

    def test_cli_archives_missing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            reports = root / "reports"
            task.write_text(json.dumps({
                "name": "cli-missing",
                "description": "",
                "command": [sys.executable, "-c", "pass"],
                "required_metrics": ["error"],
            }))
            rc = cli_main([
                "run", str(task), "--workdir", str(root), "--report-dir", str(reports),
            ])
            self.assertEqual(rc, 1)
            records = [
                json.loads(line)
                for line in (reports / "experiments.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["passed"])
            self.assertTrue(records[0]["reasons"])
            self.assertTrue(any("report" in reason for reason in records[0]["reasons"]))
            copy = Path(records[0]["acceptance_path"])
            self.assertTrue(copy.is_file())
            archived = json.loads(copy.read_text())
            self.assertEqual(archived["decision"]["reasons"], records[0]["reasons"])

    def test_python_task_uses_harness_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = MethodSpec(
                name="interpreter",
                description="",
                command=("python3", "-c", "import sys; print(sys.executable)"),
            )
            run = run_candidate(spec, directory)
            self.assertTrue(run.ok)
            self.assertEqual(run.command[0], sys.executable)

    def test_timeout_is_classified(self):
        spec = MethodSpec(
            name="timeout",
            description="",
            command=(sys.executable, "-c", "import time; time.sleep(1)"),
            timeout_s=0.01,
        )
        with tempfile.TemporaryDirectory() as directory:
            run = run_candidate(spec, directory)
        self.assertEqual(run.failure_kind, "timeout")
        self.assertFalse(run.ok)

    def test_failure_categories(self):
        self.assertEqual(classify_failure("", "SyntaxError: bad"), "syntax_error")
        self.assertEqual(classify_failure("", "AssertionError: mismatch"), "numerical_failure")
        self.assertEqual(classify_failure("", "SCF did not converge"), "scf_failure")

    def test_benchmark_summary(self):
        counter = iter(range(10))
        summary = benchmark(lambda: next(counter), warmups=1, repeats=3)
        self.assertEqual(len(summary.samples_s), 3)
        self.assertLessEqual(summary.minimum_s, summary.median_s)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_gate_reports_missing_and_bad_metrics(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        decision = AcceptanceGate(
            correctness_tolerances={"error": 1e-6}, required_metrics=("error",)
        ).evaluate(run, MetricReport({"error": 1e-3, "all_finite": False}))
        self.assertFalse(decision.passed)
        self.assertGreaterEqual(len(decision.reasons), 2)

    def test_water_cluster_atom_counts(self):
        shapes = {16: (2, 2, 4), 32: (4, 2, 4), 64: (4, 4, 4), 128: (4, 4, 8)}
        self.assertEqual(CLUSTER_COUNTS, (16, 32, 64, 128))
        for count, shape in shapes.items():
            atoms = cluster_atoms(count)
            self.assertEqual(cluster_shape(count), shape)
            self.assertEqual(len(atoms), 3 * count)
            self.assertEqual(sum(symbol == "O" for symbol, *_ in atoms), count)
            self.assertEqual(atoms[0], ("O", 0.0, 0.0, 0.0))
            self.assertEqual(atoms[1], ("H", 0.0, -0.757, 0.587))
            self.assertEqual(atoms[2], ("H", 0.0, 0.757, 0.587))
            oxygens = [atom for atom in atoms if atom[0] == "O"]
            self.assertEqual(
                {(round(x, 6), round(y, 6), round(z, 6)) for _, x, y, z in oxygens},
                {
                    (ix * 4.0, iy * 4.0, iz * 4.0)
                    for ix in range(shape[0])
                    for iy in range(shape[1])
                    for iz in range(shape[2])
                },
            )

    def test_wrong_water_cluster_nao_is_rejected(self):
        monomer_nao = 58
        correct = {count: monomer_nao * count for count in CLUSTER_COUNTS}
        self.assertEqual(nao_mismatch_max(correct, monomer_nao), 0)
        wrong = dict(correct)
        wrong[32] = 1
        mismatch = nao_mismatch_max(wrong, monomer_nao)
        self.assertGreater(mismatch, 0)
        metrics = {f"nao_{count}": wrong[count] for count in CLUSTER_COUNTS}
        metrics.update({"nao_mismatch_max": mismatch, "all_finite": True})
        decision = AcceptanceGate(
            correctness_tolerances={"nao_mismatch_max": 0},
            required_metrics=("nao_16", "nao_32", "nao_64", "nao_128", "nao_mismatch_max"),
        ).evaluate(CandidateRun("completed", ["x"], 0, 0.0, "", ""), MetricReport(metrics))
        self.assertFalse(decision.passed)
        self.assertTrue(any("nao_mismatch_max" in reason for reason in decision.reasons))

    def test_water_cluster_jk_estimate_skips_only_larger_sizes(self):
        self.assertEqual(estimate_jk_seconds(10.0, 20.0, 50.0), 25.0)
        self.assertIsNone(estimate_jk_seconds(None, 20.0, 50.0))
        self.assertTrue(should_skip_jk(25.0, 24.0))
        self.assertFalse(should_skip_jk(25.0, 25.0))
        task = json.loads(
            (ROOT / "harness/tasks/water_cluster_ccpvtz_jk.json").read_text(encoding="utf-8")
        )
        spec = MethodSpec.from_dict(task)
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.performance, {})
        self.assertEqual([case["molecules"] for case in spec.cases], [16, 32, 64, 128])
        self.assertNotIn("min_speedup", spec.performance)

    def test_water_cluster_eigh_estimate_skips_only_larger_sizes(self):
        self.assertEqual(estimate_eigh_seconds(2.0, 928, 1856), 16.0)
        self.assertEqual(estimate_eigh_seconds(2.0, 928, 7424), 1024.0)
        self.assertIsNone(estimate_eigh_seconds(None, 928, 1856))
        self.assertIsNone(estimate_eigh_seconds(0.0, 928, 1856))
        self.assertTrue(should_skip_eigh(16.1, 16.0))
        self.assertFalse(should_skip_eigh(16.0, 16.0))
        self.assertFalse(should_skip_eigh(None, 0.0))
        matrix = np.eye(2)
        energy, coeff = np.linalg.eigh(matrix)
        self.assertLess(generalized_residual(matrix, matrix, energy, coeff), 1e-12)
        task = json.loads(
            (ROOT / "harness/tasks/water_cluster_ccpvtz_guess_eigh.json").read_text(
                encoding="utf-8"
            )
        )
        spec = MethodSpec.from_dict(task)
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.performance, {})
        self.assertEqual([case["molecules"] for case in spec.cases], [16, 32, 64, 128])
        self.assertIn("guess_128_s", spec.required_metrics)
        self.assertIn("eigh_16_s", spec.required_metrics)
        self.assertNotIn("eigh_128_s", spec.required_metrics)
        self.assertNotIn("min_speedup", spec.performance)

    def test_linear_get_jk_rejects_a_quadratic_step(self):
        self.assertEqual(estimate_linear_seconds(10.0, 100, 200), 20.0)
        self.assertIsNone(estimate_linear_seconds(None, 100, 200))
        self.assertIsNone(estimate_linear_seconds(0.0, 100, 200))
        self.assertTrue(should_skip_linear(20.1, 20.0))
        self.assertFalse(should_skip_linear(20.0, 20.0))
        self.assertFalse(should_skip_linear(None, 0.0))
        self.assertAlmostEqual(scaling_exponent(10.0, 100, 20.0, 200), 1.0)
        self.assertAlmostEqual(scaling_exponent(10.0, 100, 40.0, 200), 2.0)
        self.assertIsNone(scaling_exponent(10.0, 200, 20.0, 100))
        task = json.loads(
            (ROOT / "harness/tasks/linear_get_jk.json").read_text(encoding="utf-8")
        )
        spec = MethodSpec.from_dict(task)
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.performance, {})
        self.assertEqual(spec.correctness_tolerances["exponent_water"], 1.5)
        self.assertEqual(spec.correctness_tolerances["exponent_benzene"], 1.5)
        self.assertEqual(spec.correctness_tolerances["exponent_alkane"], 1.5)
        for name in ("exponent_water", "exponent_benzene", "exponent_alkane"):
            self.assertIn(name, spec.required_metrics)
        self.assertNotIn("min_speedup", spec.performance)
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        quadratic = AcceptanceGate(
            correctness_tolerances={"exponent_alkane": 1.5},
            required_metrics=("exponent_alkane",),
        ).evaluate(run, MetricReport({"exponent_alkane": 2.0, "all_finite": True}))
        self.assertFalse(quadratic.passed)
        linear = AcceptanceGate(
            correctness_tolerances={"exponent_alkane": 1.5},
            required_metrics=("exponent_alkane",),
        ).evaluate(run, MetricReport({"exponent_alkane": 1.2, "all_finite": True}))
        self.assertTrue(linear.passed)

    def test_scf_iteration_projection_is_recorded_not_gated(self):
        self.assertEqual(TARGET_BASIS_FUNCTIONS, 50000)
        self.assertEqual(TARGET_SCF_ITERATIONS, 20)
        self.assertEqual(TARGET_WALL_HOURS, 48.0)
        self.assertAlmostEqual(
            project_iteration_hours(10.0, 100, 1.0, target_nao=200, iterations=20),
            400.0 / 3600.0,
        )
        self.assertAlmostEqual(
            project_iteration_hours(8.0, 100, 3.0, target_nao=200, iterations=1),
            64.0 / 3600.0,
        )
        self.assertIsNone(project_iteration_hours(None, 100, 1.0))
        self.assertIsNone(project_iteration_hours(10.0, 100, None))
        self.assertIsNone(project_iteration_hours(0.0, 100, 1.0))
        self.assertIsNone(project_iteration_hours(10.0, 0, 1.0))
        overlap = np.eye(2)
        density = np.array([[2.0, 0.5], [0.5, 0.0]])
        self.assertAlmostEqual(density_electrons(overlap, density), 2.0)
        self.assertEqual(density_asymmetry(density), 0.0)
        self.assertTrue(should_skip_linear(3600.1, 3600.0))
        self.assertFalse(should_skip_linear(3600.0, 3600.0))
        task = json.loads(
            (ROOT / "harness/tasks/linear_scf_iteration.json").read_text(encoding="utf-8")
        )
        spec = MethodSpec.from_dict(task)
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.performance, {})
        candidate_text = (ROOT /
            "harness_engineering/examples/linear_scf_iteration_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("training_hf", candidate_text)
        self.assertIn("iteration_rijcosx", candidate_text)
        for name in (
            "exponent_j_water", "exponent_j_benzene", "exponent_j_alkane",
            "exponent_k_water", "exponent_k_benzene", "exponent_k_alkane",
        ):
            self.assertEqual(spec.correctness_tolerances[name], 1.5)
            self.assertIn(name, spec.required_metrics)
        self.assertEqual(spec.correctness_tolerances["density_asymmetry_max"], 1e-8)
        self.assertEqual(spec.correctness_tolerances["density_electron_abs_max"], 1e-8)
        for name in (
            "exponent_eigh_water", "exponent_eigh_benzene", "exponent_eigh_alkane",
            "projected_j_hours_alkane", "projected_k_hours_alkane",
            "projected_eigh_hours_alkane",
        ):
            self.assertNotIn(name, spec.correctness_tolerances)
            self.assertNotIn(name, spec.required_metrics)
        within = {
            "nao_16": 928,
            "all_finite": True,
            "ri_j_16_max_abs": 1e-11,
            "j_parallel_16_max_abs": 0.0,
            "k_parallel_16_max_abs": 0.0,
            "j_asymmetry_16": 0.0,
            "k_asymmetry_16": 0.0,
            "jk_asymmetry_max": 0.0,
            "cosx_monomer_max_abs": 1e-4,
            "cosx_full_tensor_calls": 0,
            "density_asymmetry_max": 0.0,
            "density_electron_abs_max": 0.0,
            "projected_j_hours_alkane": 82.0,
            "projected_k_hours_alkane": 40.0,
            "projected_eigh_hours_alkane": 99.0,
        }
        for name in spec.required_metrics:
            if name.startswith("exponent_"):
                within[name] = 1.2
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        accepted = AcceptanceGate(
            correctness_tolerances=spec.correctness_tolerances,
            required_metrics=spec.required_metrics,
        ).evaluate(run, MetricReport(within))
        self.assertTrue(accepted.passed, accepted.reasons)
        quadratic = dict(within)
        quadratic["exponent_j_alkane"] = 2.0
        rejected = AcceptanceGate(
            correctness_tolerances=spec.correctness_tolerances,
            required_metrics=spec.required_metrics,
        ).evaluate(run, MetricReport(quadratic))
        self.assertFalse(rejected.passed)

    def test_iteration_hours_exponent_gate_and_timeout_skip(self):
        """Pure numbers: 50000-basis hours, the 1.5 exponent gate, and the skip."""

        alkane_jk_hours = project_iteration_hours(418.0, 2812, 1.24)
        eigh_hours = project_iteration_hours(58.4, 7424, 3.0)
        self.assertAlmostEqual(
            alkane_jk_hours,
            TARGET_SCF_ITERATIONS * 418.0
            * (TARGET_BASIS_FUNCTIONS / 2812) ** 1.24 / 3600.0,
        )
        self.assertAlmostEqual(
            project_iteration_hours(418.0, 2812, 1.24, iterations=1),
            4.119244723903265,
            places=6,
        )
        self.assertAlmostEqual(eigh_hours, 99.11433361587227, places=6)
        self.assertGreater(alkane_jk_hours, TARGET_WALL_HOURS)
        self.assertGreater(eigh_hours, TARGET_WALL_HOURS)
        self.assertIsNone(project_iteration_hours(1e300, 1, 1000.0))
        self.assertAlmostEqual(scaling_exponent(10.0, 100, 40.0, 200), 2.0)
        self.assertLess(scaling_exponent(10.0, 100, 20.0, 200), 1.5)

        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        within_gate = {
            "exponent_j_alkane": 1.5,
            "exponent_k_alkane": 1.24,
            "exponent_eigh_alkane": 3.0,
            "projected_j_hours_alkane": alkane_jk_hours,
            "projected_eigh_hours_alkane": eigh_hours,
            "all_finite": True,
        }
        accepted = AcceptanceGate(
            correctness_tolerances={
                "exponent_j_alkane": 1.5,
                "exponent_k_alkane": 1.5,
            },
            required_metrics=("exponent_j_alkane", "exponent_k_alkane"),
        ).evaluate(run, MetricReport(within_gate))
        self.assertTrue(accepted.passed, accepted.reasons)
        quadratic = dict(within_gate)
        quadratic["exponent_j_alkane"] = scaling_exponent(10.0, 100, 40.0, 200)
        rejected = AcceptanceGate(
            correctness_tolerances={
                "exponent_j_alkane": 1.5,
                "exponent_k_alkane": 1.5,
            },
            required_metrics=("exponent_j_alkane", "exponent_k_alkane"),
        ).evaluate(run, MetricReport(quadratic))
        self.assertFalse(rejected.passed)
        self.assertTrue(any("exponent_j_alkane" in reason for reason in rejected.reasons))

        # The ladder skip is linear in basis-function count and uses a 3600 s budget.
        # An equal estimate still runs. A doubled 2000 s rung does not.
        self.assertAlmostEqual(estimate_linear_seconds(2000.0, 2812, 5624), 4000.0)
        self.assertTrue(should_skip_linear(
            estimate_linear_seconds(2000.0, 2812, 5624), 3600.0))
        self.assertAlmostEqual(estimate_linear_seconds(1800.0, 1000, 2000), 3600.0)
        self.assertFalse(should_skip_linear(
            estimate_linear_seconds(1800.0, 1000, 2000), 3600.0))
        self.assertFalse(should_skip_linear(3600.0, 3600.0))
        self.assertTrue(should_skip_linear(3600.0 + 1e-6, 3600.0))
        self.assertIsNone(estimate_linear_seconds(None, 2812, 5624))
        self.assertFalse(should_skip_linear(None, 3600.0))
        cubic = 58.4 * (14848 / 7424) ** 3
        self.assertAlmostEqual(estimate_linear_seconds(58.4, 7424, 14848), 116.8)
        self.assertGreater(cubic, 116.8)
        self.assertFalse(should_skip_linear(
            estimate_linear_seconds(58.4, 7424, 14848), 3600.0))

    def test_generation_round_trip_and_stop_rules(self):
        raw = {
            "name": "occupy-density-oracle",
            "description": "generation",
            "command": [sys.executable, "-c", "pass"],
            "generation": {
                "editable": ["harness_engineering/generated/occupy_density.py"],
                "forbidden": ["hf.py"],
                "stop": {"syntax_fixes": 1, "stall_rounds": 2},
            },
        }
        spec = MethodSpec.from_dict(raw)
        self.assertEqual(spec.to_dict()["generation"]["editable"], raw["generation"]["editable"])
        self.assertEqual(spec.to_dict()["generation"]["forbidden"], ["hf.py"])
        self.assertEqual(MethodSpec.from_dict({
            "name": "plain",
            "command": [sys.executable, "-c", "pass"],
        }).generation, {})
        task = json.loads(
            (ROOT / "harness/tasks/occupy_density_budget.json").read_text(encoding="utf-8")
        )
        budget = MethodSpec.from_dict(task)
        self.assertEqual(budget.timeout_s, 3600)
        self.assertEqual(budget.correctness_tolerances["projected_eigh_hours_benzene"], 48)
        self.assertIn("projected_eigh_hours_water", budget.required_metrics)
        self.assertIn("../hf.py", budget.generation["forbidden"])
        self.assertNotIn("../hf.py", budget.generation["editable"])
        self.assertTrue(is_projection_reason(
            "projected_eigh_hours_benzene=117.955 exceeds tolerance 48"))
        self.assertFalse(is_projection_reason("density_max_abs=0.1 exceeds tolerance 1e-08"))
        reasons = [
            "projected_eigh_hours_water=112.041 exceeds tolerance 48",
            "projected_eigh_hours_benzene=117.955 exceeds tolerance 48",
            "projected_eigh_hours_alkane=94.479 exceeds tolerance 48",
        ]
        self.assertTrue(correct_despite_projection(reasons))
        kept = decide_round(
            oracle_passed=True,
            budget_passed=False,
            budget_reasons=reasons,
            projection_value=117.955,
        )
        self.assertEqual(kept["action"], "keep_correct_candidate")
        self.assertTrue(kept["correct_candidate"])
        self.assertTrue(kept["run_budget"])
        stalled = decide_round(
            oracle_passed=True,
            budget_passed=False,
            budget_reasons=reasons,
            projection_history=(117.955, 117.955),
            projection_value=118.0,
        )
        self.assertEqual(stalled["action"], "stop")
        self.assertTrue(stalled["correct_candidate"])
        self.assertFalse(projections_stalled((118.0, 100.0)))
        self.assertTrue(projections_stalled((118.0, 118.0, 120.0)))
        blocked = decide_round(
            oracle_passed=False,
            oracle_reasons=["density_max_abs=0.2 exceeds tolerance 1e-08"],
        )
        self.assertEqual(blocked["action"], "revert")
        self.assertFalse(blocked["run_budget"])
        self.assertFalse(blocked["correct_candidate"])
        first_fix = decide_round(
            oracle_passed=False,
            oracle_failure_kind="syntax_error",
            oracle_reasons=["syntax_error"],
            syntax_fixes=0,
        )
        self.assertEqual(first_fix["action"], "fix_once")
        self.assertFalse(first_fix["run_budget"])
        second_fix = decide_round(
            oracle_passed=False,
            oracle_failure_kind="syntax_error",
            oracle_reasons=["syntax_error"],
            syntax_fixes=1,
        )
        self.assertEqual(second_fix["action"], "stop_before_budget")
        accepted = decide_round(oracle_passed=True, budget_passed=True)
        self.assertEqual(accepted["action"], "accept")
        fock = np.diag([1.0, 3.0, 2.0])
        overlap = np.eye(3)
        density = occupy_density_reference(fock, overlap, 1)
        self.assertAlmostEqual(density_electrons(overlap, density), 2.0)
        self.assertAlmostEqual(density[0, 0], 2.0)
        self.assertEqual(density_asymmetry(density), 0.0)

    def test_performance_gate_requires_its_metric(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        decision = AcceptanceGate(max_slowdown=1.25).evaluate(
            run, MetricReport({"all_finite": True})
        )
        self.assertFalse(decision.passed)
        self.assertIn("missing performance metric: slowdown", decision.reasons)

    def test_gate_rejects_nonfinite_and_non_numeric_correctness(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        gate = AcceptanceGate(correctness_tolerances={"error": 1e-7})
        for value in (float("nan"), float("inf"), -float("inf"),
                      False, True, "0", None, [], {}):
            with self.subTest(value=value):
                result = gate.evaluate(run, MetricReport({"error": value}))
                self.assertFalse(result.passed)
        self.assertTrue(gate.evaluate(run, MetricReport({"error": 0.0})).passed)

    def test_gate_rejects_nonfinite_and_non_numeric_performance(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        for metric, gate in (("speedup", AcceptanceGate(min_speedup=1.05)),
                             ("slowdown", AcceptanceGate(max_slowdown=1.25))):
            for value in (float("nan"), float("inf"), -float("inf"),
                          False, True, "1.1", None, [], {}):
                with self.subTest(metric=metric, value=value):
                    result = gate.evaluate(run, MetricReport({metric: value}))
                    self.assertFalse(result.passed)
            self.assertTrue(gate.evaluate(run, MetricReport({metric: 1.1})).passed)
            self.assertFalse(gate.evaluate(run, MetricReport({})).passed)

    def test_gate_rejects_nonfinite_ungated_measurements(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        result = AcceptanceGate().evaluate(
            run, MetricReport({"median_seconds": float("nan"), "all_finite": True}))
        self.assertFalse(result.passed)

    def test_gate_scientific_success_flags_must_be_true_booleans(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        for metric in ("all_finite", "scf_converged"):
            for value in (False, 0, 1, "true", None):
                with self.subTest(metric=metric, value=value):
                    self.assertFalse(AcceptanceGate().evaluate(
                        run, MetricReport({metric: value})).passed)
            self.assertTrue(AcceptanceGate().evaluate(
                run, MetricReport({metric: True})).passed)

    def test_gate_rejects_invalid_thresholds(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        for value in (float("nan"), float("inf"), -1.0, False, "0"):
            with self.subTest(threshold=value):
                self.assertFalse(AcceptanceGate(
                    correctness_tolerances={"error": value}).evaluate(
                        run, MetricReport({"error": 0.0})).passed)
                self.assertFalse(AcceptanceGate(min_speedup=value).evaluate(
                    run, MetricReport({"speedup": 1.1})).passed)
                self.assertFalse(AcceptanceGate(max_slowdown=value).evaluate(
                    run, MetricReport({"slowdown": 1.1})).passed)


if __name__ == "__main__":
    unittest.main()
