import os
import time

import requests

from agents.base_agent import AgentResult, BaseAgent


class SalesMindAdapter(BaseAgent):
    name = "salesmind"

    def __init__(
        self,
        base_url: str | None = None,
        session_id: str | None = None,
        timeout: float = 120.0,
        transport=None,
    ):
        self.base_url = (base_url or os.environ.get("SALESMIND_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.session_id = session_id or os.environ.get("SALESMIND_SESSION_ID", "")
        self.timeout = timeout
        self._transport = transport

    def run(self, user_input: str) -> AgentResult:
        start = time.perf_counter()
        session_id = self.session_id or self._create_session()
        response = self._post(
            f"{self.base_url}/api/sessions/{session_id}/chat",
            {"message": user_input},
        )
        latency = time.perf_counter() - start
        return self._to_result(response, latency)

    def _create_session(self) -> str:
        payload = self._post(f"{self.base_url}/api/sessions", {})
        session_id = str(payload.get("id", "") or "")
        if not session_id:
            raise RuntimeError("SalesMind did not return a session id")
        self.session_id = session_id
        return session_id

    def _post(self, url: str, payload: dict) -> dict:
        if self._transport is not None:
            return self._transport(url, payload)
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _to_result(response: dict, latency: float) -> AgentResult:
        message = response.get("message", {})
        answer = str(message.get("content", "") or "")
        trace = response.get("trace", []) or []
        tool_calls = []
        for step in trace:
            for call in step.get("toolCalls", []) or []:
                tool_calls.append(
                    {
                        "name": str(call.get("name", "") or ""),
                        "arguments": call.get("details", {}) or {},
                    }
                )
        metadata = {
            "agent_used": response.get("agentUsed"),
            "confidence": response.get("confidence"),
            "sources": response.get("sources"),
            "trace": trace,
            "follow_up_tasks": response.get("followUpTasks"),
        }
        return AgentResult(
            answer=answer,
            tool_calls=tool_calls,
            trajectory=[call["name"] for call in tool_calls],
            latency=round(latency, 4),
            metadata=metadata,
        )
