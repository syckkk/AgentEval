from evaluation.llm_judge import LLMJudge


class FailureAnalyzer:
    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge

    def analyze_results(self, results: list[dict]) -> list[dict]:
        return [self.analyze(result) for result in results if result.get("status") == "FAIL"]

    def analyze(self, evaluated: dict) -> dict:
        if evaluated.get("status") == "PASS":
            return {"case_id": evaluated.get("case_id", ""), "status": "PASS"}

        rule_result = self._rule_analysis(evaluated)
        if rule_result["failure_type"] != "UNKNOWN":
            return rule_result

        if self.judge is not None:
            llm_result = self.judge.analyze_failure(evaluated)
            if llm_result.get("failure_type") != "UNKNOWN":
                return llm_result
        return rule_result

    def _rule_analysis(self, evaluated: dict) -> dict:
        case_id = str(evaluated.get("case_id", ""))
        metrics = evaluated.get("metrics", {})
        tool = metrics.get("tool_correctness", {})
        expected_tools = tool.get("expected", [])
        actual_tools = tool.get("actual", [])

        if not tool.get("pass", False):
            if set(expected_tools) == set(actual_tools):
                return self._trajectory_error(case_id, expected_tools, actual_tools)
            return self._tool_selection_error(case_id, expected_tools, actual_tools, tool)

        argument_issue = self._argument_issue(evaluated)
        if argument_issue:
            return self._tool_argument_error(case_id, argument_issue)

        task = metrics.get("task_success", {})
        answer = metrics.get("answer_quality", {})
        if not task.get("pass", False) or not answer.get("pass", False):
            return self._answer_quality_error(case_id, evaluated, task, answer)

        return self._unknown(case_id)

    @staticmethod
    def _argument_issue(evaluated: dict) -> dict | None:
        case = evaluated.get("case", {})
        expected_specs = case.get("expected_arguments", [])
        actual_calls = evaluated.get("agent_result", {}).get("tool_calls", [])
        for spec in expected_specs:
            name = str(spec.get("name", ""))
            expected_args = spec.get("arguments", {})
            call = next((item for item in actual_calls if item.get("name") == name), None)
            if call is None:
                continue
            if call.get("arguments", {}) != expected_args:
                return {
                    "tool": name,
                    "expected": expected_args,
                    "actual": call.get("arguments", {}),
                }
        return None

    @staticmethod
    def _tool_selection_error(case_id: str, expected_tools: list, actual_tools: list, tool: dict) -> dict:
        return {
            "case_id": case_id,
            "status": "FAIL",
            "failure_type": "TOOL_SELECTION_ERROR",
            "expected": {"tools": expected_tools},
            "actual": {"tools": actual_tools},
            "reason": f"Expected tools {expected_tools} but got {actual_tools}.",
            "suggestion": "Check tool descriptions and the agent prompt.",
        }

    @staticmethod
    def _trajectory_error(case_id: str, expected_tools: list, actual_tools: list) -> dict:
        return {
            "case_id": case_id,
            "status": "FAIL",
            "failure_type": "TRAJECTORY_ERROR",
            "expected": {"trajectory": expected_tools},
            "actual": {"trajectory": actual_tools},
            "reason": "Tools are correct but called in the wrong order.",
            "suggestion": "Check the agent workflow and step planning rules.",
        }

    @staticmethod
    def _tool_argument_error(case_id: str, issue: dict) -> dict:
        return {
            "case_id": case_id,
            "status": "FAIL",
            "failure_type": "TOOL_ARGUMENT_ERROR",
            "expected": {"arguments": {issue["tool"]: issue["expected"]}},
            "actual": {"arguments": {issue["tool"]: issue["actual"]}},
            "reason": f"Tool '{issue['tool']}' was correct but its arguments did not match.",
            "suggestion": "Check argument extraction and tool parameter descriptions.",
        }

    @staticmethod
    def _answer_quality_error(case_id: str, evaluated: dict, task: dict, answer: dict) -> dict:
        agent_result = evaluated.get("agent_result", {})
        return {
            "case_id": case_id,
            "status": "FAIL",
            "failure_type": "ANSWER_QUALITY_ERROR",
            "expected": {"answer": task.get("expected_answer", "") or answer.get("expected_answer", "")},
            "actual": {"answer": agent_result.get("answer", "")},
            "reason": "Tool usage looks correct but the final answer does not satisfy the expected answer.",
            "suggestion": "Check the final answer prompt and grounding of tool results.",
        }

    @staticmethod
    def _unknown(case_id: str) -> dict:
        return {
            "case_id": case_id,
            "status": "FAIL",
            "failure_type": "UNKNOWN",
            "reason": "Rule-based analysis could not determine the failure type.",
            "suggestion": "Review the trace manually or use LLM fallback.",
        }
