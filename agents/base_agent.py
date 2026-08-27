from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[str] = field(default_factory=list)
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "trajectory": self.trajectory,
            "latency": self.latency,
            "metadata": self.metadata,
        }


class BaseAgent:
    name = "base"

    def run(self, user_input: str) -> AgentResult:
        raise NotImplementedError
