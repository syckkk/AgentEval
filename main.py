import argparse
import json
from pathlib import Path

import yaml

from agents.demo_agent import DemoAgent
from agents.salesmind_adapter import SalesMindAdapter
from analysis.failure_analyzer import FailureAnalyzer
from evaluation.llm_judge import LLMJudge
from regression.regression_tester import RegressionTester
from reports.report_generator import generate_regression_report_html
from runner.evaluation_runner import EvaluationRunner

PROJECT_ROOT = Path(__file__).resolve().parent


def load_cases(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["test_cases"] if isinstance(payload, dict) else payload


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_judge(config: dict) -> LLMJudge | None:
    section = config.get("judge") or {}
    if not section.get("enabled"):
        return None
    return LLMJudge(
        model=section.get("model", "deepseek-chat"),
        base_url=section.get("base_url", "https://api.deepseek.com/v1"),
        api_key=section.get("api_key", ""),
    )


def load_evaluation(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_regression(args) -> None:
    baseline_payload = load_evaluation(args.baseline)
    current_payload = load_evaluation(args.current)
    result = RegressionTester().compare(
        baseline_payload.get("results", []),
        current_payload.get("results", []),
    )

    out_path = Path(args.out or str(PROJECT_ROOT / "results" / "regression" / "regression_report.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    html_path = Path(args.html or str(PROJECT_ROOT / "results" / "regression" / "regression_report.html"))
    generate_regression_report_html(result, html_path)
    print(f"status={result['status']} overall_change={result['overall_score_change']}")
    print(f"saved: {out_path}")
    print(f"saved html: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentEval - evaluate an agent against test cases.")
    parser.add_argument("command", choices=["evaluate", "regression"])
    parser.add_argument("--agent", default="demo", help="agent name, currently only demo")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "test_cases.json"))
    parser.add_argument("--baseline", default=str(PROJECT_ROOT / "results" / "baseline" / "evaluation.json"))
    parser.add_argument("--current", default=str(PROJECT_ROOT / "results" / "current" / "evaluation.json"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--html", default=None)
    parser.add_argument("--out-failures", default=str(PROJECT_ROOT / "results" / "current" / "failure_analysis.json"))
    parser.add_argument("--limit", type=int, default=0, help="limit number of cases to run")
    args = parser.parse_args()

    if args.command == "regression":
        run_regression(args)
        return

    config = load_config()
    if args.agent == "demo":
        agent = DemoAgent()
    elif args.agent == "salesmind":
        salesmind = config.get("salesmind") or {}
        agent = SalesMindAdapter(
            base_url=salesmind.get("base_url") or None,
            session_id=salesmind.get("session_id") or None,
        )
    else:
        parser.error(f"Unknown agent: {args.agent}")

    judge = build_judge(config)
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    runner = EvaluationRunner(agent, judge=judge)
    results = runner.run_cases(cases)
    out_path = Path(args.out or str(PROJECT_ROOT / "results" / "current" / "evaluation.json"))
    path = runner.save(results, out_path)
    analyzer = FailureAnalyzer(judge=judge)
    failures = analyzer.analyze_results(results)
    failure_path = runner.save_failures(failures, args.out_failures)

    passed = sum(1 for result in results if result["status"] == "PASS")
    average = round(sum(result["overall_score"] for result in results) / len(results), 4) if results else 0.0
    print(f"agent={agent.name} cases={len(results)} pass={passed}/{len(results)} overall_avg={average}")
    print(f"saved: {path}")
    print(f"saved failures: {failure_path}")


if __name__ == "__main__":
    main()
