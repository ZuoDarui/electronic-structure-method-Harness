import json
from pathlib import Path
import sys
import tempfile
import unittest

from harness_engineering.benchmark import benchmark, percentile
from harness_engineering.models import AcceptanceGate, CandidateRun, MethodSpec, MetricReport
from harness_engineering.cli import main as cli_main
from harness_engineering.runner import classify_failure, evaluate_candidate, run_candidate
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
            Path("harness/tasks/water_cluster_ccpvtz_jk.json").read_text(encoding="utf-8")
        )
        spec = MethodSpec.from_dict(task)
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.performance, {})
        self.assertEqual([case["molecules"] for case in spec.cases], [16, 32, 64, 128])
        self.assertNotIn("min_speedup", spec.performance)

    def test_performance_gate_requires_its_metric(self):
        run = CandidateRun("completed", ["x"], 0, 0.0, "", "")
        decision = AcceptanceGate(max_slowdown=1.25).evaluate(
            run, MetricReport({"all_finite": True})
        )
        self.assertFalse(decision.passed)
        self.assertIn("missing performance metric: slowdown", decision.reasons)


if __name__ == "__main__":
    unittest.main()
