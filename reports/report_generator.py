from datetime import datetime
from html import escape
from pathlib import Path

STATUS_COLORS = {
    "IMPROVED": "#6af7a0",
    "UNCHANGED": "#8a8aa0",
    "REGRESSED": "#f76a6a",
    "REGRESSION DETECTED": "#f0c040",
}


def generate_regression_report_html(regression_result: dict, output_path: str | Path) -> Path:
    status = str(regression_result.get("status", "UNKNOWN"))
    color = STATUS_COLORS.get(status, "#8a8aa0")

    rows = []
    for name, item in regression_result.get("metric_comparison", {}).items():
        change = float(item.get("change", 0.0))
        sign = "+" if change > 0 else ""
        rows.append(
            f"""<tr>
              <td>{escape(name)}</td>
              <td>{float(item.get('baseline', 0.0)):.1%}</td>
              <td>{float(item.get('current', 0.0)):.1%}</td>
              <td>{sign}{change:.1%}</td>
            </tr>"""
        )

    baseline_overall = regression_result.get("baseline", {}).get("overall_score", 0.0)
    current_overall = regression_result.get("current", {}).get("overall_score", 0.0)
    overall_change = regression_result.get("overall_score_change", 0.0)
    overall_sign = "+" if overall_change > 0 else ""
    critical = regression_result.get("critical_regression", {})
    critical_notes = "<br>".join(
        f"<b>{escape(name)}</b>: {'REGRESSED' if regressed else 'OK'}"
        for name, regressed in critical.items()
    ) or "none"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentEval Regression Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #f6f7f9; color: #1c1c22; padding: 32px; }}
  h1 {{ font-size: 24px; margin: 0 0 8px; }}
  .meta {{ color: #6b6b76; font-size: 13px; margin-bottom: 20px; }}
  .status {{ display: inline-block; padding: 6px 14px; border-radius: 999px; color: {color}; background: {color}18; border: 1px solid {color}66; font-weight: 700; }}
  .overall {{ margin: 18px 0; font-size: 15px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 760px; background: #fff; }}
  th, td {{ border: 1px solid #d9dce1; padding: 10px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f0f1f4; }}
</style>
</head>
<body>
  <h1>AgentEval Regression Report</h1>
  <div class="meta">Generated: {datetime.now().strftime("%d %b %Y %H:%M")}</div>
  <div><span class="status">{escape(status)}</span></div>
  <div class="overall">
    Baseline overall: <b>{baseline_overall}</b>
    &nbsp;|&nbsp; Current overall: <b>{current_overall}</b>
    &nbsp;|&nbsp; Change: <b>{overall_sign}{overall_change}</b>
  </div>
  <table>
    <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
    {''.join(rows)}
  </table>
  <div style="margin-top:20px;font-size:14px">Critical metrics:<br>{critical_notes}</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
