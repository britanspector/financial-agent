"""Run the fixed offline Phase 5.3 context-selection and retrieval baseline."""

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
from financial_agent.config import Settings
from financial_agent.context.runtime import build_context_manager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed offline context-selection evaluation")
    parser.add_argument("--last-n", type=int, default=3, help="Turn count for last_n")
    parser.add_argument("--budget-tokens", type=int, default=160, help="History budget for context strategies")
    parser.add_argument("--json-report", type=Path, help="Write the complete machine-readable report")
    parser.add_argument("--live-summary", action="store_true", help="Use configured Qwen for summary_compression")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    cases = load_context_eval_cases(root / "eval" / "context" / "context_cases.jsonl")
    manager = None
    if args.live_summary:
        settings = Settings(context_strategy="summary_compression")
        manager = build_context_manager(settings=settings)
    report = evaluate_context_selection(
        cases,
        manager=manager,
        last_n=args.last_n,
        budget_tokens=args.budget_tokens,
    )
    print(json.dumps(report_summary(report), ensure_ascii=False, indent=2))
    if args.json_report:
        write_json_report(report, args.json_report)
    if not report.summary_baseline_passed or not report.retrieval_baseline_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
