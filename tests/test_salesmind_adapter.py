import unittest

from agents.salesmind_adapter import SalesMindAdapter
from runner.evaluation_runner import EvaluationRunner


def fake_transport(url: str, payload: dict) -> dict:
    if url.endswith("/api/sessions"):
        return {"id": "session-1"}
    return {
        "message": {"content": "OpenFit 2售价为1299元"},
        "agentUsed": "question",
        "confidence": 0.9,
        "sources": [],
        "trace": [
            {
                "agent": "rag",
                "status": "completed",
                "toolCalls": [
                    {
                        "name": "search_product",
                        "details": {"product": "OpenFit 2"},
                        "success": True,
                    }
                ],
            }
        ],
    }


class SalesMindAdapterTest(unittest.TestCase):
    def test_parses_chat_response(self):
        adapter = SalesMindAdapter(session_id="existing-session", transport=fake_transport)
        result = adapter.run("OpenFit 2多少钱？")
        self.assertEqual(result.answer, "OpenFit 2售价为1299元")
        self.assertEqual(result.trajectory, ["search_product"])
        self.assertEqual(result.tool_calls[0]["arguments"], {"product": "OpenFit 2"})

    def test_creates_session_when_missing(self):
        adapter = SalesMindAdapter(transport=fake_transport)
        result = adapter.run("OpenFit 2多少钱？")
        self.assertEqual(adapter.session_id, "session-1")
        self.assertEqual(result.trajectory, ["search_product"])

    def test_no_tool_calls(self):
        transport = lambda url, payload: {
            "message": {"content": "你好"},
            "agentUsed": "question",
            "trace": [],
        }
        adapter = SalesMindAdapter(session_id="existing-session", transport=transport)
        result = adapter.run("你好")
        self.assertEqual(result.trajectory, [])
        self.assertEqual(result.tool_calls, [])

    def test_works_with_evaluation_engine_without_changes(self):
        adapter = SalesMindAdapter(session_id="existing-session", transport=fake_transport)
        cases = [
            {
                "id": "TC001",
                "input": "OpenFit 2多少钱？",
                "expected_tools": ["search_product"],
                "expected_answer": "OpenFit 2售价为1299元",
            }
        ]
        runner = EvaluationRunner(adapter)
        results = runner.run_cases(cases)
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(results[0]["overall_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
