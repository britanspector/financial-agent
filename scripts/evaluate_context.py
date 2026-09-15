"""Run the fixed offline Phase 5.1 context-selection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_agent.context.evaluation import (
    evaluate_context_selection,
    load_context_eval_cases,
    report_summary,
    write_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed offline context-selection evaluation")
    parser.add_argument("--last-n", type=int, default=3, help="Turn count for last_n")
    parser.add_argument("--budget-tokens", type=int, default=130, help="History budget for budgeted_selection")
    parser.add_argument("--json-report", type=Path, help="Write the complete machine-readable report")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    cases = load_context_eval_cases(root / "eval" / "context" / "context_cases.jsonl")
    report = evaluate_context_selection(
        cases,
        last_n=args.last_n,
        budget_tokens=args.budget_tokens,
    )
    print(json.dumps(report_summary(report), ensure_ascii=False, indent=2))
    if args.json_report:
        write_json_report(report, args.json_report)


if __name__ == "__main__":
    main()
