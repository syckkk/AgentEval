import json
import tempfile
import unittest
from pathlib import Path

from agents.demo_agent import DemoAgent
from runner.evaluation_runner import EvaluationRunner


class EvaluationRunnerTest(unittest.TestCase):
    def test_end_to_end_saves_json(self):
        cases = [
            {"id": "TC001", "input": "OpenFit 2多少钱？", "expected_tools": ["search_product"], "expected_answer": "OpenFit 2售价为1299元"},
            {"id": "TC002", "input": "你好", "expected_tools": [], "expected_answer": "你好"},
            {"id": "TC003", "input": "耳机多少钱？", "expected_tools": [], "expected_answer": "哪款耳机"},
        ]
        agent = DemoAgent()
        runner = EvaluationRunner(agent)
        results = runner.run_cases(cases)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(results[1]["status"], "PASS")
        self.assertEqual(results[2]["status"], "PASS")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evaluation.json"
            path = runner.save(results, output)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_cases"], 3)
            self.assertEqual(payload["agent"], "demo")
            self.assertIn("agent_result", payload["results"][0])


if __name__ == "__main__":
    unittest.main()
