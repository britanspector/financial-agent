"""Run the fixed offline Phase 6.1 Agent E2E evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_agent.agent.e2e_evaluation import evaluate_agent_e2e, load_e2e_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic Agent E2E evaluation")
    parser.add_argument("--json-report", type=Path, help="Write the complete machine-readable report")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    scenarios, expectations = load_e2e_eval_set(
        root / "eval" / "agent_e2e" / "scenarios.jsonl",
        root / "eval" / "agent_e2e" / "expectations.jsonl",
    )
    report = evaluate_agent_e2e(scenarios, expectations)
    print(json.dumps({
        "case_count": report.case_count,
        "passed_count": report.passed_count,
        "all_passed": report.all_passed,
        "metrics": report.metrics.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2))
    if args.json_report:
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
