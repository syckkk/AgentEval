import json
from pathlib import Path

from evaluation.evaluator import EvaluationEngine


class EvaluationRunner:
    def __init__(self, agent, judge=None):
        self.agent = agent
        self.judge = judge
        self.engine = EvaluationEngine(judge=judge)

    def run_cases(self, cases: list[dict]) -> list[dict]:
        results = []
        for case in cases:
            agent_result = self.agent.run(case.get("input", ""))
            evaluated = self.engine.evaluate(case, agent_result)
            evaluated["input"] = case.get("input", "")
            evaluated["case"] = case
            evaluated["agent_result"] = agent_result.to_dict()
            results.append(evaluated)
        return results

    def save(self, results: list[dict], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": self.agent.name,
            "judge": {"model": self.judge.model, "base_url": self.judge.base_url} if self.judge else None,
            "total_cases": len(results),
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    def save_failures(self, failures: list[dict], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": self.agent.name,
            "total_failures": len(failures),
            "failures": failures,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path
