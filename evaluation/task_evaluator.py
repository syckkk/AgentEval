import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


class TaskEvaluator:
    name = "task_success"

    def __init__(self, metric_name: str = "task_success"):
        self.name = metric_name

    def evaluate(self, case: dict, agent_result) -> dict:
        expected = str(case.get("expected_answer", "") or "").strip()
        actual = str(agent_result.answer or "").strip()
        if not expected:
            return {"pass": True, "score": 1.0, "reason": "No expected answer defined."}
        passed = normalize_text(expected) in normalize_text(actual)
        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "expected_answer": expected,
            "actual_answer": actual,
            "reason": "Expected answer found in actual answer." if passed else "Expected answer not found in actual answer.",
        }
