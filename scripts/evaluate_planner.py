"""Run Planner Eval v0.1 against the configured Qwen planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_agent.config import Settings
from financial_agent.knowledge.runtime import register_rag_tools
from financial_agent.market_data.runtime import register_market_tools
from financial_agent.planner.evaluation import (
    PlannerEvaluationReport,
    PlannerEvalMetrics,
    evaluate_planner_with_report,
    load_planner_eval_cases,
)
from financial_agent.planner.runtime import build_planner
from financial_agent.tools.composite import merge_registries
from financial_agent.user_data.runtime import register_user_tools


class _UnusedService:
    def execute(self, *args, **kwargs):
        raise RuntimeError("Planner evaluation does not execute tools")


def _markdown_report(report) -> str:
    lines = ["# Planner Eval Case Report", "", "## Metrics", "", "```json",
             report.metrics.model_dump_json(indent=2), "```", "", "## Cases", ""]
    for case in report.cases:
        lines.extend([
            f"### {case.case_id}", "",
            f"Query: {case.query}", "",
            "Expected plan:", "", "```json",
            json.dumps(case.expected_plan.model_dump(mode="json"), ensure_ascii=False, indent=2), "```", "",
            "Actual plan:", "", "```json",
            case.actual_plan.model_dump_json(indent=2), "```", "",
            f"- validation: `{case.validation_valid}`",
            f"- tool / argument / temporal / dependency: `{case.tool_correct}` / `{case.argument_correct}` / `{case.temporal_correct}` / `{case.dependency_correct}`",
            f"- reasons: {'; '.join(case.error_reasons)}", "",
        ])
        for detail in case.argument_details:
            if detail.status != "exact":
                lines.append(f"  - argument `{detail.tool_name}` ({detail.status}): {detail.reason}; expected={detail.expected!r}; actual={detail.actual!r}")
        lines.append("")
    return "\n".join(lines)


def _combine_reports(paths: list[Path]) -> PlannerEvaluationReport:
    reports = [PlannerEvaluationReport.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]
    cases = [case for report in reports for case in report.cases]
    executable = [case for case in cases if case.expectation == "executable"]
    argument_total = sum(len(case.argument_details) for case in cases)
    temporal_total = sum(case.temporal_total_count for case in cases)
    planned_total = sum(case.planned_task_count for case in cases)
    return PlannerEvaluationReport(
        metrics=PlannerEvalMetrics(
            case_count=len(cases),
            executable_case_count=len(executable),
            abstention_case_count=len(cases) - len(executable),
            valid_plan_rate=sum(case.validation_valid for case in executable) / len(executable),
            tool_selection_accuracy=sum(case.tool_correct for case in cases) / len(cases),
            argument_accuracy=sum(detail.correct for case in cases for detail in case.argument_details) / argument_total,
            temporal_accuracy=sum(case.temporal_correct_count for case in cases) / temporal_total,
            dependency_accuracy=sum(case.dependency_score for case in cases) / len(cases),
            unnecessary_tool_call_rate=sum(case.unnecessary_calls for case in cases) / planned_total,
        ),
        cases=cases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Planner Eval against configured Qwen")
    parser.add_argument("--report", type=Path, help="Write a per-case Markdown report")
    parser.add_argument("--json-report", type=Path, help="Write a machine-readable per-case report")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first case index")
    parser.add_argument("--end", type=int, help="Exclusive case index")
    parser.add_argument("--combine-json", type=Path, nargs="+", help="Combine prior batch JSON reports without a model call")
    args = parser.parse_args()
    if args.combine_json:
        report = _combine_reports(args.combine_json)
    else:
        settings = Settings()
        service = _UnusedService()
        catalog = merge_registries(
            register_user_tools(service),
            register_market_tools(service),
            register_rag_tools(service),
        )
        planner, validator = build_planner(settings, catalog)
        case_path = Path(__file__).parents[1] / "eval" / "planner" / "planner_cases.jsonl"
        cases = load_planner_eval_cases(case_path)
        report = evaluate_planner_with_report(cases[args.start:args.end], planner, validator)
    print(report.metrics.model_dump_json(indent=2))
    if args.report:
        args.report.write_text(_markdown_report(report), encoding="utf-8")
    if args.json_report:
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
