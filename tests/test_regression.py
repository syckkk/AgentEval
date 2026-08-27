import json
import tempfile
import unittest
from pathlib import Path

from regression.regression_tester import RegressionTester
from reports.report_generator import generate_regression_report_html


def make_result(
    overall: float,
    task_pass: bool = True,
    tool_pass: bool = True,
    traj_pass: bool = True,
    answer_pass: bool = True,
) -> dict:
    return {
        "overall_score": overall,
        "metrics": {
            "task_success": {"pass": task_pass, "score": 1.0 if task_pass else 0.0},
            "tool_correctness": {"pass": tool_pass, "score": 1.0 if tool_pass else 0.0},
            "trajectory_match": {"pass": traj_pass, "score": 1.0 if traj_pass else 0.0},
            "answer_quality": {"pass": answer_pass, "score": 1.0 if answer_pass else 0.0},
        },
    }


class RegressionTesterTest(unittest.TestCase):
    def test_improved(self):
        baseline = [make_result(0.5) for _ in range(2)]
        current = [make_result(1.0) for _ in range(2)]
        result = RegressionTester().compare(baseline, current)
        self.assertEqual(result["status"], "IMPROVED")
        self.assertGreater(result["overall_score_change"], 0)

    def test_regressed(self):
        baseline = [make_result(1.0) for _ in range(2)]
        current = [make_result(0.5) for _ in range(2)]
        result = RegressionTester().compare(baseline, current)
        self.assertEqual(result["status"], "REGRESSED")

    def test_unchanged(self):
        baseline = [make_result(0.8) for _ in range(2)]
        current = [make_result(0.8) for _ in range(2)]
        result = RegressionTester().compare(baseline, current)
        self.assertEqual(result["status"], "UNCHANGED")

    def test_critical_regression_detected_despite_overall_improvement(self):
        baseline = [
            make_result(0.6, task_pass=True, tool_pass=False),
            make_result(0.6, task_pass=True, tool_pass=False),
        ]
        current = [
            make_result(0.8, task_pass=False, tool_pass=True),
            make_result(0.8, task_pass=False, tool_pass=True),
        ]
        result = RegressionTester().compare(baseline, current)
        self.assertGreater(result["overall_score_change"], 0)
        self.assertEqual(result["status"], "REGRESSION DETECTED")
        self.assertTrue(result["critical_regression"]["task_success"])

    def test_compare_files(self):
        baseline = [make_result(0.5)]
        current = [make_result(1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            current_path = Path(tmp) / "current.json"
            baseline_path.write_text(json.dumps({"results": baseline}), encoding="utf-8")
            current_path.write_text(json.dumps({"results": current}), encoding="utf-8")
            result = RegressionTester().compare_files(baseline_path, current_path)
            self.assertEqual(result["status"], "IMPROVED")

    def test_html_report(self):
        result = RegressionTester().compare([make_result(0.5)], [make_result(1.0)])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            path = generate_regression_report_html(result, output)
            html = path.read_text(encoding="utf-8")
            self.assertIn("IMPROVED", html)
            self.assertIn("task_success", html)


if __name__ == "__main__":
    unittest.main()
