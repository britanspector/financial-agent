"""Run the Phase 5.4 offline or live context ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.config import Settings
from financial_agent.context.ablation import evaluate_context_ablation, load_ablation_cases
from financial_agent.context.qwen_summary_provider import QwenSummaryProvider
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.verifier.qwen_provider import QwenVerifierProvider


LIVE_CASE_IDS = {
    "holdout_portfolio_identity",
    "holdout_assistant_cashflow",
    "holdout_corrected_date",
    "holdout_constraint_reversal",
    "holdout_long_recent_noise",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.4 Context Manager ablation")
    parser.add_argument("--live", action="store_true", help="Use Qwen for the fixed five-case live slice")
    parser.add_argument("--budget-tokens", type=int, default=160)
    parser.add_argument("--last-n", type=int, default=3)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    path = ROOT / "eval" / "context_ablation" / "holdout_cases.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = load_ablation_cases(path)
    factories = None
    mode = "live" if args.live else "offline"
    if args.live:
        cases = [case for case in cases if case.case_id in LIVE_CASE_IDS]
        settings = Settings()
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required for live context ablation")
        key = settings.qwen_api_key.get_secret_value()
        factories = (
            lambda: QwenSummaryProvider(
                key, model=settings.summary_model, base_url=settings.summary_base_url,
                timeout=settings.summary_timeout_seconds, temperature=0,
            ),
            lambda: QwenPlannerProvider(
                key, model=settings.planner_model, base_url=settings.planner_base_url,
                timeout=settings.planner_timeout_seconds, temperature=0,
            ),
            lambda: QwenAnswerProvider(
                key, model=settings.answer_model, base_url=settings.answer_base_url,
                timeout=settings.answer_timeout_seconds, temperature=0,
            ),
            lambda: QwenVerifierProvider(
                key, model=settings.verifier_model, base_url=settings.verifier_base_url,
                timeout=settings.verifier_timeout_seconds, temperature=0,
            ),
        )
    report = evaluate_context_ablation(
        cases, mode=mode, budget_tokens=args.budget_tokens, last_n=args.last_n,
        live_factories=factories, holdout_sha256=digest,
    )
    summary = {
        "mode": report.mode,
        "holdout_sha256": report.holdout_sha256,
        "case_count": report.case_count,
        "hard_gate_passed": report.hard_gate_passed,
        "hard_gate_failures": report.hard_gate_failures,
        "experimental_targets": report.experimental_targets,
        "strategies": [item.model_dump(mode="json") for item in report.strategies],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not report.hard_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
