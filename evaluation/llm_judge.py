import json
import os
import re

import requests


class LLMJudge:
    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str = "",
        call_fn=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._call_fn = call_fn

    def judge_answer(self, user_input: str, expected_answer: str, actual_answer: str, criteria: dict) -> dict:
        prompt = self._answer_prompt(user_input, expected_answer, actual_answer, criteria or {})
        raw = self._request(prompt, "You are a strict but fair answer quality judge. Always respond with valid JSON only.")
        parsed = self._parse_json(raw)
        if parsed is None:
            return {
                "score": 0.0,
                "pass": False,
                "reason": "Judge returned invalid JSON.",
                "raw_judge_output": raw[:300],
            }
        return self._normalize_answer(parsed, raw)

    def judge_trajectory(self, case: dict, agent_result) -> dict:
        prompt = self._trajectory_prompt(case, agent_result)
        raw = self._request(prompt, "You are a strict but fair trajectory judge. Always respond with valid JSON only.")
        parsed = self._parse_json(raw)
        if parsed is None:
            return {
                "overall_score": 0.0,
                "verdict": "fail",
                "summary": "Judge returned invalid JSON.",
                "raw_judge_output": raw[:300],
            }
        return self._normalize_trajectory(parsed, raw)

    def analyze_failure(self, evaluated: dict) -> dict:
        prompt = self._failure_prompt(evaluated)
        raw = self._request(prompt, "You are a strict failure analysis expert. Always respond with valid JSON only.")
        parsed = self._parse_json(raw)
        if parsed is None:
            return {
                "failure_type": "UNKNOWN",
                "reason": "Judge returned invalid JSON.",
                "suggestion": "Review the trace manually.",
            }
        failure_type = str(parsed.get("failure_type", "UNKNOWN")).upper().replace("-", "_").replace(" ", "_")
        allowed = {"TOOL_SELECTION_ERROR", "TOOL_ARGUMENT_ERROR", "TRAJECTORY_ERROR", "ANSWER_QUALITY_ERROR"}
        if failure_type not in allowed:
            failure_type = "UNKNOWN"
        return {
            "failure_type": failure_type,
            "reason": str(parsed.get("reason", "") or ""),
            "suggestion": str(parsed.get("suggestion", "") or ""),
            "raw_judge_output": raw,
        }

    def _request(self, prompt: str, system: str) -> str:
        if self._call_fn is not None:
            return self._call_fn(prompt, system)
        if not self.api_key:
            return ""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        text = re.sub(r"<think>[\s\S]*?</think>", "", str(raw or ""), flags=re.IGNORECASE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _normalize_answer(parsed: dict, raw: str) -> dict:
        try:
            score = float(parsed.get("score", 0.0))
            passed = bool(parsed.get("pass", score >= 0.7))
        except (TypeError, ValueError):
            score, passed = 0.0, False
        score = max(0.0, min(1.0, score))
        return {
            "score": round(score, 4),
            "pass": passed,
            "reason": str(parsed.get("reason", "") or ""),
            "raw_judge_output": raw,
        }

    @staticmethod
    def _normalize_trajectory(parsed: dict, raw: str) -> dict:
        overall = parsed.get("overall", {})
        try:
            score = float(overall.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        verdict = str(overall.get("verdict", "fail")).lower()
        return {
            "tool_selection_score": parsed.get("tool_selection", {}).get("score"),
            "argument_quality_score": parsed.get("argument_quality", {}).get("score"),
            "answer_groundedness_score": parsed.get("answer_groundedness", {}).get("score"),
            "stop_appropriateness_score": parsed.get("stop_appropriateness", {}).get("score"),
            "overall_score": score,
            "verdict": verdict,
            "passed": verdict == "pass",
            "summary": str(overall.get("summary", "") or ""),
            "raw_judge_output": raw,
        }

    @staticmethod
    def _answer_prompt(user_input: str, expected_answer: str, actual_answer: str, criteria: dict) -> str:
        return f"""You are evaluating whether an agent completed a user request.

USER INPUT: {user_input}
EXPECTED ANSWER: {expected_answer}
ACTUAL ANSWER: {actual_answer}
EVALUATION CRITERIA: {json.dumps(criteria, ensure_ascii=False)}

Score the actual answer 0.0-1.0. Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "pass": <true or false>, "reason": "<one sentence>"}}
Pass means the answer satisfies the expected intent."""

    @staticmethod
    def _trajectory_prompt(case: dict, agent_result) -> str:
        steps = []
        for idx, call in enumerate(agent_result.tool_calls, 1):
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            steps.append(f"Step {idx}: TOOL CALL - {call.get('name')} args={args}")
        if not steps:
            steps.append("No tool calls.")
        trajectory = "\n".join(steps)
        return f"""You are a trajectory evaluator for an AI agent.

QUESTION: {case.get('input', '')}
EXPECTED BEHAVIOR: {case.get('expected_answer', '')}
EXPECTED TOOL SEQUENCE: {case.get('expected_tools', [])}
ACTUAL TOOL SEQUENCE: {agent_result.trajectory}
FINAL ANSWER: {agent_result.answer}

TRAJECTORY:
{trajectory}

Respond ONLY with valid JSON:
{{
  "tool_selection": {{"score": 1-5, "reason": "<one sentence>"}},
  "argument_quality": {{"score": 1-5, "reason": "<one sentence>"}},
  "answer_groundedness": {{"score": 1-5, "reason": "<one sentence>"}},
  "stop_appropriateness": {{"score": 1-5, "reason": "<one sentence>"}},
  "overall": {{"score": 1-5, "verdict": "<pass or fail>", "summary": "<two sentences max>"}}
}}
Verdict is pass when overall >= 3."""

    @staticmethod
    def _failure_prompt(evaluated: dict) -> str:
        metrics = evaluated.get("metrics", {})
        agent_result = evaluated.get("agent_result", {})
        return f"""You are analyzing why an agent evaluation case failed.

CASE ID: {evaluated.get('case_id', '')}
USER INPUT: {evaluated.get('input', '')}
METRICS: {json.dumps(metrics, ensure_ascii=False, indent=2)}
AGENT RESULT: {json.dumps(agent_result, ensure_ascii=False, indent=2)}

The rule-based analysis could not classify the failure.
Choose exactly one failure type and respond ONLY with valid JSON:
{{"failure_type": "TOOL_SELECTION_ERROR|TOOL_ARGUMENT_ERROR|TRAJECTORY_ERROR|ANSWER_QUALITY_ERROR", "reason": "<one sentence>", "suggestion": "<one sentence>"}}"""
