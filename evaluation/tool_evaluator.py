class ToolEvaluator:
    name = "tool_correctness"

    def evaluate(self, case: dict, agent_result) -> dict:
        expected = [str(t).strip() for t in case.get("expected_tools", []) if str(t).strip()]
        actual = [str(t).strip() for t in agent_result.trajectory if str(t).strip()]
        passed = expected == actual
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong_order = (not passed) and set(expected) == set(actual)
        if passed:
            reason = "Tool sequence exactly matched."
        elif wrong_order:
            reason = "Tools match but order is wrong."
        else:
            reason = "Tool selection differs from expected."
        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "expected": expected,
            "actual": actual,
            "missing_tools": missing,
            "unexpected_tools": unexpected,
            "wrong_order": wrong_order,
            "reason": reason,
        }
