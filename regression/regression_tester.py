import json
from pathlib import Path
from statistics import mean

ALL_METRICS = ["task_success", "tool_correctness", "trajectory_match", "answer_quality"]
DEFAULT_CRITICAL_METRICS = ["task_success", "tool_correctness"]


def _metric_stats(results: list[dict], metric_name: str) -> dict:
    values = [result.get("metrics", {}).get(metric_name, {}) for result in results]
    if not values:
        return {"pass_rate": 0.0, "average_score": 0.0}
    pass_rate = sum(1 for value in values if value.get("pass")) / len(values)
    average_score = mean(float(value.get("score", 0.0)) for value in values)
    return {"pass_rate": round(pass_rate, 4), "average_score": round(average_score, 4)}


def _overall_score(results: list[dict]) -> float:
    if not results:
        return 0.0
    return round(mean(float(result.get("overall_score", 0.0)) for result in results), 4)


class RegressionTester:
    def __init__(self, critical_metrics: list[str] | None = None):
        self.critical_metrics = critical_metrics or list(DEFAULT_CRITICAL_METRICS)

    def compare(self, baseline_results: list[dict], current_results: list[dict]) -> dict:
        baseline_stats = {name: _metric_stats(baseline_results, name) for name in ALL_METRICS}
        current_stats = {name: _metric_stats(current_results, name) for name in ALL_METRICS}
        baseline_overall = _overall_score(baseline_results)
        current_overall = _overall_score(current_results)

        comparison = {}
        for name in ALL_METRICS:
            comparison[name] = {
                "baseline": baseline_stats[name]["pass_rate"],
                "current": current_stats[name]["pass_rate"],
                "change": round(current_stats[name]["pass_rate"] - baseline_stats[name]["pass_rate"], 4),
                "baseline_average_score": baseline_stats[name]["average_score"],
                "current_average_score": current_stats[name]["average_score"],
            }

        critical_regression = {
            name: comparison[name]["current"] < comparison[name]["baseline"]
            for name in self.critical_metrics
        }
        any_critical_regression = any(critical_regression.values())

        if any_critical_regression:
            status = "REGRESSION DETECTED"
        elif current_overall > baseline_overall:
            status = "IMPROVED"
        elif current_overall < baseline_overall:
            status = "REGRESSED"
        else:
            status = "UNCHANGED"

        return {
            "baseline": {"overall_score": baseline_overall, "metrics": baseline_stats},
            "current": {"overall_score": current_overall, "metrics": current_stats},
            "metric_comparison": comparison,
            "overall_score_change": round(current_overall - baseline_overall, 4),
            "critical_metrics": self.critical_metrics,
            "critical_regression": critical_regression,
            "any_critical_regression": any_critical_regression,
            "status": status,
        }

    @staticmethod
    def _load_results(path: str | Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("results", [])

    def compare_files(self, baseline_path: str | Path, current_path: str | Path) -> dict:
        return self.compare(self._load_results(baseline_path), self._load_results(current_path))
