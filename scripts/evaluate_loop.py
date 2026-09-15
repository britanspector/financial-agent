"""Run the fixed offline Phase 4.3 Loop Eval."""

from __future__ import annotations

import argparse
from pathlib import Path

from financial_agent.agent.loop_evaluation import evaluate_loop_cases, load_loop_eval_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed offline bounded-loop evaluation")
    parser.add_argument("--json-report", type=Path, help="Write the complete machine-readable report")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    report = evaluate_loop_cases(load_loop_eval_cases(root / "eval" / "loop" / "loop_cases.jsonl"))
    summary = {"case_count": report.case_count, "passed_count": report.passed_count,
               "all_passed": report.all_passed}
    print(json_dumps(summary))
    if args.json_report:
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not report.all_passed:
        raise SystemExit(1)


def json_dumps(value: object) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
