"""Run the frozen Phase 6.4 Qwen holdout against deterministic Tool fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from financial_agent.agent.live_evaluation import (
    LiveEvaluationReport, evaluate_live_agent, load_live_eval_set,
    rescore_live_failure_attribution,
)
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.config import Settings
from financial_agent.context.qwen_summary_provider import QwenSummaryProvider
from financial_agent.observability import JsonlTraceSink
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.verifier.qwen_provider import QwenVerifierProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 6.4 live Agent evaluation")
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--rescore-report", type=Path,
                        help="Correct failure attribution from a saved report without Qwen calls")
    args = parser.parse_args()
    eval_set = load_live_eval_set(ROOT / "eval" / "live_agent_e2e" / "frozen_config.json")
    if args.rescore_report:
        existing = LiveEvaluationReport.model_validate_json(
            args.rescore_report.read_text(encoding="utf-8")
        )
        if existing.manifest_sha256 != eval_set.manifest_sha256:
            raise ValueError("Saved report does not match the frozen live manifest")
        report = rescore_live_failure_attribution(existing, eval_set)
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({
            "manifest_sha256": report.manifest_sha256,
            "rescored_without_model_calls": True,
            "scorer_schema_fixes_after_freeze": report.scorer_schema_fixes_after_freeze,
            "aggregates": [item.model_dump(mode="json") for item in report.aggregates],
            "comparison": report.comparison.model_dump(mode="json"),
        }, ensure_ascii=False, indent=2))
        return
    if args.trace_jsonl is None:
        parser.error("--trace-jsonl is required for a live run")
    settings = Settings()
    if settings.qwen_api_key is None:
        raise ValueError("FINANCIAL_AGENT_QWEN_API_KEY is required for the live evaluation")
    key = settings.qwen_api_key.get_secret_value()
    model = eval_set.frozen_config.model

    def provider_factory(component, case_id, configuration, run_attempt):
        del case_id, configuration, run_attempt
        common = {"model": model, "temperature": 0}
        if component == "planner":
            return QwenPlannerProvider(
                key, base_url=settings.planner_base_url,
                timeout=settings.planner_timeout_seconds, **common,
            )
        if component == "writer":
            return QwenAnswerProvider(
                key, base_url=settings.answer_base_url,
                timeout=settings.answer_timeout_seconds, **common,
            )
        if component == "verifier":
            return QwenVerifierProvider(
                key, base_url=settings.verifier_base_url,
                timeout=settings.verifier_timeout_seconds, **common,
            )
        if component == "summary":
            return QwenSummaryProvider(
                key, base_url=settings.summary_base_url,
                timeout=settings.summary_timeout_seconds, **common,
            )
        raise ValueError(f"Unknown live provider component: {component}")

    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_live_agent(
        eval_set, provider_factory, trace_sink=JsonlTraceSink(args.trace_jsonl),
    )
    args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest_sha256": report.manifest_sha256,
        "case_count": report.case_count,
        "run_attempt_count": report.run_attempt_count,
        "harness_hard_gate_passed": report.harness_hard_gate_passed,
        "hard_gate_failures": report.hard_gate_failures,
        "aggregates": [item.model_dump(mode="json") for item in report.aggregates],
        "comparison": report.comparison.model_dump(mode="json"),
        "json_report": str(args.json_report),
        "trace_jsonl": str(args.trace_jsonl),
    }, ensure_ascii=False, indent=2))
    if not report.harness_hard_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
