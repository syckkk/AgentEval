from evaluation.task_evaluator import TaskEvaluator


class AnswerQualityEvaluator:
    name = "answer_quality"

    def __init__(self, judge=None):
        self.judge = judge
        self._fallback = TaskEvaluator(metric_name="answer_quality")

    def evaluate(self, case: dict, agent_result) -> dict:
        if self.judge is not None:
            return self.judge.judge_answer(
                user_input=case.get("input", ""),
                expected_answer=case.get("expected_answer", ""),
                actual_answer=agent_result.answer,
                criteria=case.get("evaluation_criteria", {}),
            )
        return self._fallback.evaluate(case, agent_result)
