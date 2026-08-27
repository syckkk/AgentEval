import unittest

from analysis.failure_analyzer import FailureAnalyzer
from evaluation.llm_judge import LLMJudge


def evaluated_case(
    case_id: str,
    expected_tools: list,
    actual_tools: list,
    task_pass: bool = True,
    answer_pass: bool = True,
    expected_arguments: list | None = None,
    tool_calls: list | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "status": "FAIL",
        "input": "test input",
        "case": {"expected_arguments": expected_arguments or []},
        "metrics": {
            "tool_correctness": {
                "pass": expected_tools == actual_tools,
                "expected": expected_tools,
                "actual": actual_tools,
            },
            "task_success": {"pass": task_pass, "expected_answer": "expected"},
            "answer_quality": {"pass": answer_pass},
        },
        "agent_result": {
            "answer": "actual",
            "tool_calls": tool_calls or [{"name": name, "arguments": {}} for name in actual_tools],
        },
    }


class FailureAnalyzerTest(unittest.TestCase):
    def test_pass_case(self):
        result = FailureAnalyzer().analyze({"case_id": "TC001", "status": "PASS"})
        self.assertEqual(result["status"], "PASS")

    def test_tool_selection_error(self):
        result = FailureAnalyzer().analyze(
            evaluated_case("TC002", ["search_product"], ["calculate_price"])
        )
        self.assertEqual(result["failure_type"], "TOOL_SELECTION_ERROR")

    def test_trajectory_error(self):
        result = FailureAnalyzer().analyze(
            evaluated_case("TC003", ["search_product", "check_inventory"], ["check_inventory", "search_product"])
        )
        self.assertEqual(result["failure_type"], "TRAJECTORY_ERROR")

    def test_tool_argument_error(self):
        result = FailureAnalyzer().analyze(
            evaluated_case(
                "TC004",
                ["search_product"],
                ["search_product"],
                expected_arguments=[{"name": "search_product", "arguments": {"product": "OpenFit 2"}}],
                tool_calls=[{"name": "search_product", "arguments": {"product": "OpenFit"}}],
            )
        )
        self.assertEqual(result["failure_type"], "TOOL_ARGUMENT_ERROR")

    def test_answer_quality_error(self):
        result = FailureAnalyzer().analyze(
            evaluated_case("TC005", ["search_product"], ["search_product"], task_pass=False)
        )
        self.assertEqual(result["failure_type"], "ANSWER_QUALITY_ERROR")

    def test_unknown_without_judge(self):
        result = FailureAnalyzer().analyze(
            evaluated_case("TC006", [], [], task_pass=True, answer_pass=True)
        )
        self.assertEqual(result["failure_type"], "UNKNOWN")

    def test_llm_fallback(self):
        judge = LLMJudge(
            call_fn=lambda prompt, system: (
                '{"failure_type": "ANSWER_QUALITY_ERROR", "reason": "missing price", "suggestion": "check prompt"}'
            )
        )
        result = FailureAnalyzer(judge=judge).analyze(
            evaluated_case("TC007", [], [], task_pass=True, answer_pass=True)
        )
        self.assertEqual(result["failure_type"], "ANSWER_QUALITY_ERROR")
        self.assertEqual(result["suggestion"], "check prompt")

    def test_analyze_results_skips_passes(self):
        analyzer = FailureAnalyzer()
        results = [
            {"case_id": "TC008", "status": "PASS"},
            evaluated_case("TC009", ["search_product"], ["calculate_price"]),
        ]
        failures = analyzer.analyze_results(results)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_type"], "TOOL_SELECTION_ERROR")


if __name__ == "__main__":
    unittest.main()
