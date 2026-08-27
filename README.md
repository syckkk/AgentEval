# AgentEval

AgentEval is a lightweight, explainable evaluation and regression testing
system for AI agents. It reuses the agent/trajectory evaluation ideas from
`amitbad/llm-evaluation` and adds two product capabilities on top:

1. Regression testing: compare Agent V1 vs Agent V2 on the same test set.
2. Failure analysis: classify why a case failed and suggest where to look.

## Status

Phase 1 (understand and run the original project) is complete.
Phase 2 (project skeleton) is complete.
Phase 3 (basic evaluation loop) is complete: test dataset, DemoAgent,
task/tool/trajectory evaluators, LLM-as-a-Judge, evaluation runner, and
CLI. The demo run passes 15/15 cases with an overall score of 1.0.
Phase 4 (failure analysis) is complete: rule-first classification for
tool selection, tool arguments, trajectory order, and answer quality, with
LLM fallback when rules return UNKNOWN.
Phase 5 (regression testing) is complete: baseline vs current comparison,
critical metric checks, and JSON + HTML regression reports.
Phase 6 (SalesMind adapter) is complete: `--agent salesmind` calls the
SalesMind backend chat API and maps `message.content` plus
`trace[].toolCalls` into AgentEval's `AgentResult` without changing the
evaluation core.

## Run

```bash
python -m unittest discover -s tests -v
python main.py evaluate
```

The evaluate command writes `results/current/evaluation.json`. Set
`judge.enabled: true` in `config.yaml` and provide `DEEPSEEK_API_KEY` (or
another OpenAI-compatible key) to activate LLM-as-a-Judge.

The same command also writes `results/current/failure_analysis.json`
containing only failed cases, each with `failure_type`, `reason`, and
`suggestion`.

Regression:

```bash
python main.py evaluate --out results/baseline/evaluation.json
python main.py evaluate --out results/current/evaluation.json
python main.py regression
```

The regression command compares baseline and current evaluation results and
writes `results/regression/regression_report.json` plus
`results/regression/regression_report.html`. Overall improvement is the
default signal, but a drop in any critical metric (`task_success`,
`tool_correctness`) overrides the verdict with `REGRESSION DETECTED`.

SalesMind (requires the SalesMind backend running on port 4000):

```bash
python main.py evaluate --agent salesmind --limit 3
```

`SALESMIND_BASE_URL` and `SALESMIND_SESSION_ID` environment variables can
override the values in `config.yaml`.

## Phase 1 findings

The original repository is kept at the workspace root for reference:

```text
phase-5-production-eval/
  agent-evaluation/test12_agent_evaluator.py    # deterministic agent eval
  trajectory-evaluation/test13_trajectory_evaluator.py  # judge-scored eval
```

Test 12 inputs:

- `agent_test_cases.xlsx`: 15 cases with question, expected tool path,
  expected end state, account ID, search query hints, step limit, and why
  the case exists.
- `config.yaml` -> `phase5_agent_evaluation`: bot model, provider, Ollama
  URL, max steps.
- `agent.py` + `tools.py`: prompt-driven tool-calling agent with
  `lookup_account_status`, `search_docs`, and `calculator`.
- `mock_account_data.json` and `documents/`: deterministic tool backends.

Test 12 output:

- `results/agent_evaluation_results.json`: per case, expected vs actual tool
  sequence, account ID and search query, stop reason, final answer, full
  trace, and boolean checks (`tool_match`, `argument_match`,
  `stop_quality_match`).
- `results/agent_evaluation_report.html`: human-readable report.

Test 12 core logic:

- Tool path check: exact sequence match of expected tool names.
- Argument check: account ID exact match; search query passes when at least
  half of the expected hint words appear.
- Stop check: final action matches expected end, no premature stop, no
  `max_steps_reached` or malformed output.

Test 13 inputs:

- Test 12 results JSON (traces).
- `config.yaml` -> `phase5_trajectory_evaluation`: judge model/provider.

Test 13 output:

- `results/trajectory_evaluation_results.json`: per case, four dimensions
  scored 1-5 (tool selection, argument quality, answer groundedness, stop
  appropriateness), overall score, verdict, reasons, and raw judge output.
- `results/trajectory_evaluation_report.html`.

Test 13 core logic:

- Builds a judge prompt from the question, expected behavior, expected tool
  sequence, actual tool sequence, trace steps, stop reason, and final answer.
- Requires valid JSON output, with a regex fallback for wrapped JSON.
- Verdict: `pass` when overall score >= 3.

## Local Phase 1 run

Ollama was not available on this machine, so the original scripts were run
against DeepSeek (`deepseek-chat`) using the OpenAI-compatible provider path
through a local proxy. Results were written by the original scripts without
modifying their evaluation logic.

Test 12 (DeepSeek, 2026-08-27):

```text
15/15 cases ran
tool_match:          13/15
argument_match:      13/15
stop_quality_match:  13/15
unnecessary_tool_use: 0
```

TC09 and TC10 are multi-step cases: the agent called
`lookup_account_status` but skipped the expected `search_docs` step and
stopped with a final answer.

Test 13 (DeepSeek judge, 2026-08-27, on the fresh Test 12 traces):

```text
15/15 trajectories scored
15/15 pass
avg overall:            4.87
avg tool selection:     4.73
avg argument quality:   4.93
avg answer groundedness:4.93
avg stop appropriateness:4.67
```

TC09 scored 4/5 and TC10 scored 4/5; both lost points on tool selection and
stop appropriateness because the expected second tool call was missing.

## Planned layout

```text
AgentEval/
  data/          test cases
  agents/        base agent interface + adapters (demo, salesmind)
  evaluation/    task, tool, trajectory evaluators + LLM judge
  runner/        evaluation runner
  analysis/      rule-first failure analysis with LLM fallback
  regression/    V1 vs V2 regression testing
  reports/       JSON + HTML report generation
  results/       baseline / current / regression outputs
  tests/         unittest suites for agents, evaluators, and runner
  main.py        CLI entry point
```
