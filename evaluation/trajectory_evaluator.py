class TrajectoryEvaluator:
    name = "trajectory_match"

    def __init__(self, judge=None):
        self.judge = judge

    def evaluate(self, case: dict, agent_result) -> dict:
        expected = [str(t).strip() for t in case.get("expected_tools", []) if str(t).strip()]
        actual = [str(t).strip() for t in agent_result.trajectory if str(t).strip()]
        passed = expected == actual
        same_set = set(expected) == set(actual)
        if passed:
            score = 1.0
            reason = "Trajectory matches the expected tool sequence."
        elif same_set:
            score = 0.5
            reason = "Trajectory uses the same tools but in a different order."
        else:
            score = 0.0
            reason = "Trajectory is missing or uses unexpected tools."
        result = {
            "pass": passed,
            "score": score,
            "expected": expected,
            "actual": actual,
            "reason": reason,
        }
        if self.judge is not None:
            judge_result = self.judge.judge_trajectory(case, agent_result)
            result["judge_score"] = judge_result.get("overall_score")
            result["judge_verdict"] = judge_result.get("verdict")
            result["judge_reason"] = judge_result.get("summary", "")
        return result
