import unittest

from agents.base_agent import AgentResult
from evaluation.evaluator import EvaluationEngine
from evaluation.llm_judge import LLMJudge
from evaluation.task_evaluator import TaskEvaluator
from evaluation.tool_evaluator import ToolEvaluator
from evaluation.trajectory_evaluator import TrajectoryEvaluator


class TaskEvaluatorTest(unittest.TestCase):
    def test_pass_and_fail(self):
        case = {"expected_answer": "售价为1299元"}
        ok = TaskEvaluator().evaluate(case, AgentResult(answer="OpenFit 2目前售价为1299元。"))
        bad = TaskEvaluator().evaluate(case, AgentResult(answer="我不知道。"))
        self.assertTrue(ok["pass"])
        self.assertFalse(bad["pass"])


class ToolEvaluatorTest(unittest.TestCase):
    def test_exact_match(self):
        case = {"expected_tools": ["search_product", "check_inventory"]}
        result = ToolEvaluator().evaluate(case, AgentResult(trajectory=["search_product", "check_inventory"]))
        self.assertTrue(result["pass"])
        self.assertEqual(result["missing_tools"], [])
        self.assertEqual(result["unexpected_tools"], [])

    def test_missing_and_unexpected(self):
        case = {"expected_tools": ["search_product"]}
        result = ToolEvaluator().evaluate(case, AgentResult(trajectory=["calculate_price"]))
        self.assertFalse(result["pass"])
        self.assertEqual(result["missing_tools"], ["search_product"])
        self.assertEqual(result["unexpected_tools"], ["calculate_price"])

    def test_wrong_order(self):
        case = {"expected_tools": ["search_product", "check_inventory"]}
        result = ToolEvaluator().evaluate(case, AgentResult(trajectory=["check_inventory", "search_product"]))
        self.assertFalse(result["pass"])
        self.assertTrue(result["wrong_order"])


class TrajectoryEvaluatorTest(unittest.TestCase):
    def test_exact(self):
        case = {"expected_tools": ["search_product", "check_inventory"]}
        result = TrajectoryEvaluator().evaluate(case, AgentResult(trajectory=["search_product", "check_inventory"]))
        self.assertTrue(result["pass"])
        self.assertEqual(result["score"], 1.0)

    def test_wrong_order_partial_score(self):
        case = {"expected_tools": ["search_product", "check_inventory"]}
        result = TrajectoryEvaluator().evaluate(case, AgentResult(trajectory=["check_inventory", "search_product"]))
        self.assertFalse(result["pass"])
        self.assertEqual(result["score"], 0.5)


class LLMJudgeTest(unittest.TestCase):
    def test_answer_judge_parses_json(self):
        judge = LLMJudge(call_fn=lambda prompt, system: '{"score": 0.9, "pass": true, "reason": "good"}')
        result = judge.judge_answer("q", "expected", "actual", {})
        self.assertTrue(result["pass"])
        self.assertEqual(result["score"], 0.9)

    def test_answer_judge_invalid_json(self):
        judge = LLMJudge(call_fn=lambda prompt, system: "not json")
        result = judge.judge_answer("q", "expected", "actual", {})
        self.assertFalse(result["pass"])
        self.assertEqual(result["score"], 0.0)

    def test_trajectory_judge_parses_json(self):
        judge = LLMJudge(
            call_fn=lambda prompt, system: (
                '{"tool_selection":{"score":5,"reason":"ok"},'
                '"argument_quality":{"score":5,"reason":"ok"},'
                '"answer_groundedness":{"score":5,"reason":"ok"},'
                '"stop_appropriateness":{"score":5,"reason":"ok"},'
                '"overall":{"score":5,"verdict":"pass","summary":"good"}}'
            )
        )
        result = judge.judge_trajectory(
            {"input": "q", "expected_tools": ["a"]},
            AgentResult(trajectory=["a"], answer="ok"),
        )
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["overall_score"], 5.0)


class EvaluationEngineTest(unittest.TestCase):
    def test_four_metrics_and_status(self):
        case = {
            "id": "TC001",
            "input": "OpenFit 2多少钱？",
            "expected_tools": ["search_product"],
            "expected_answer": "1299元",
        }
        agent_result = AgentResult(
            answer="OpenFit 2售价为1299元",
            tool_calls=[{"name": "search_product", "arguments": {"product": "OpenFit 2"}}],
            trajectory=["search_product"],
        )
        evaluated = EvaluationEngine().evaluate(case, agent_result)
        self.assertEqual(evaluated["status"], "PASS")
        self.assertEqual(
            set(evaluated["metrics"]),
            {"task_success", "tool_correctness", "trajectory_match", "answer_quality"},
        )
        self.assertEqual(evaluated["overall_score"], 1.0)

    def test_llm_judge_metric_replaces_fallback(self):
        judge = LLMJudge(call_fn=lambda prompt, system: '{"score": 0.6, "pass": false, "reason": "missing price"}')
        case = {"id": "TC002", "input": "q", "expected_tools": [], "expected_answer": "x"}
        agent_result = AgentResult(answer="x", trajectory=[])
        evaluated = EvaluationEngine(judge=judge).evaluate(case, agent_result)
        self.assertEqual(evaluated["metrics"]["answer_quality"]["score"], 0.6)


if __name__ == "__main__":
    unittest.main()
