from evaluation.answer_quality_evaluator import AnswerQualityEvaluator
from evaluation.task_evaluator import TaskEvaluator
from evaluation.tool_evaluator import ToolEvaluator
from evaluation.trajectory_evaluator import TrajectoryEvaluator


class EvaluationEngine:
    def __init__(self, judge=None):
        self.judge = judge
        self.evaluators = [
            TaskEvaluator(),
            ToolEvaluator(),
            TrajectoryEvaluator(judge=judge),
            AnswerQualityEvaluator(judge=judge),
        ]

    def evaluate(self, case: dict, agent_result) -> dict:
        metrics = {}
        for evaluator in self.evaluators:
            metrics[evaluator.name] = evaluator.evaluate(case, agent_result)
        scores = [metric.get("score", 0.0) for metric in metrics.values()]
        overall = round(sum(scores) / len(scores), 4) if scores else 0.0
        all_pass = all(metric.get("pass", False) for metric in metrics.values())
        return {
            "case_id": str(case.get("id", "")),
            "status": "PASS" if all_pass else "FAIL",
            "overall_score": overall,
            "metrics": metrics,
        }
