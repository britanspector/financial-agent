"""Fixed offline evaluation for history context selection strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from financial_agent.context.manager import ContextManager
from financial_agent.context.models import ContextMetrics, ContextPolicy, ContextStrategy
from financial_agent.schemas import Message, Schema, UserQuery


class CriticalContextItem(Schema):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    kind: Literal["entity", "constraint", "context"]
    source_role: Literal["user", "assistant"] = "user"
    planner_required: bool = True
    hard_required: bool | None = None

    @model_validator(mode="after")
    def default_hard_requirement(self):
        if self.hard_required is None:
            self.hard_required = self.kind in {"entity", "constraint"}
        return self


class ContextEvalCase(Schema):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[Message] = Field(min_length=1)
    critical_context: list[CriticalContextItem] = Field(min_length=1)

    def request(self) -> UserQuery:
        return UserQuery(query=self.query, history=self.history)


class ContextEvalObservation(Schema):
    case_id: str
    strategy: ContextStrategy
    metrics: ContextMetrics
    retained_keys: list[str]
    lost_keys: list[str]
    planner_equivalent_to_full_history: bool
    passed: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class ContextStrategyMetrics(Schema):
    strategy: ContextStrategy
    case_count: int = Field(ge=1)
    critical_context_retention_rate: float = Field(ge=0, le=1)
    history_token_compression_ratio: float = Field(ge=0, le=1)
    planner_plan_equivalence_rate: float = Field(ge=0, le=1)
    critical_entity_or_constraint_loss_rate: float = Field(ge=0, le=1)


class ContextEvaluationReport(Schema):
    case_count: int = Field(ge=1)
    planner_probe: Literal["deterministic_required_context"] = "deterministic_required_context"
    last_n: int = Field(ge=0)
    budget_tokens: int = Field(ge=0)
    strategies: list[ContextStrategyMetrics]
    cases: list[ContextEvalObservation]
    summary_baseline_passed: bool
    summary_failed_case_ids: list[str] = Field(default_factory=list)


def load_context_eval_cases(path: Path) -> list[ContextEvalCase]:
    cases: list[ContextEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(ContextEvalCase.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid context eval case at line {line_number}") from exc
    if not 10 <= len(cases) <= 15:
        raise ValueError("Context eval baseline must contain 10 to 15 cases")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Context eval case IDs must be unique")
    return cases


def evaluate_context_selection(
    cases: list[ContextEvalCase],
    *,
    manager: ContextManager | None = None,
    last_n: int = 3,
    budget_tokens: int = 130,
) -> ContextEvaluationReport:
    if not cases:
        raise ValueError("At least one context eval case is required")
    selector = manager or _offline_summary_manager()
    policies = [
        ContextPolicy(strategy="full_history", last_n=last_n, budget_tokens=budget_tokens),
        ContextPolicy(strategy="last_n", last_n=last_n, budget_tokens=budget_tokens),
        ContextPolicy(strategy="budgeted_selection", last_n=last_n, budget_tokens=budget_tokens),
        ContextPolicy(
            strategy="summary_compression", last_n=last_n, budget_tokens=budget_tokens,
            summary_recent_n=3, summary_budget_ratio=0.4,
        ),
    ]
    observations: list[ContextEvalObservation] = []
    for case in cases:
        full_signature = _planner_probe_signature(case, case.request())
        for policy in policies:
            selection = selector.select(case.request(), "planner", policy)
            retained = [
                item.key for item in case.critical_context
                if _item_is_available(item, selection.request, selection.summary, case.history)
            ]
            lost = [item.key for item in case.critical_context if item.key not in retained]
            hard_lost = [item.key for item in case.critical_context if item.hard_required and item.key in lost]
            failures = []
            if policy.strategy == "summary_compression" and hard_lost:
                failures.append("hard_required context lost: " + ", ".join(hard_lost))
            if policy.strategy == "summary_compression" and selection.metrics.selected_tokens > budget_tokens:
                failures.append("history token budget exceeded")
            observations.append(ContextEvalObservation(
                case_id=case.case_id,
                strategy=policy.strategy,
                metrics=selection.metrics,
                retained_keys=retained,
                lost_keys=lost,
                planner_equivalent_to_full_history=(
                    _planner_probe_signature(
                        case, selection.request, selection.summary, case.history,
                    ) == full_signature
                ),
                passed=not failures,
                failure_reasons=failures,
            ))

    strategy_metrics = [
        _aggregate_strategy(strategy, cases, observations)
        for strategy in ("full_history", "last_n", "budgeted_selection", "summary_compression")
    ]
    summary_metrics = next(item for item in strategy_metrics if item.strategy == "summary_compression")
    failed_case_ids = [
        item.case_id for item in observations
        if item.strategy == "summary_compression" and not item.passed
    ]
    baseline_passed = (
        not failed_case_ids
        and summary_metrics.critical_context_retention_rate >= 0.90
        and summary_metrics.planner_plan_equivalence_rate >= 0.90
        and summary_metrics.critical_entity_or_constraint_loss_rate <= 0.10
    )
    return ContextEvaluationReport(
        case_count=len(cases),
        last_n=last_n,
        budget_tokens=budget_tokens,
        strategies=strategy_metrics,
        cases=observations,
        summary_baseline_passed=baseline_passed,
        summary_failed_case_ids=failed_case_ids,
    )


def _item_is_available(item, request, summary=None, original_history=None) -> bool:
    if item.value in request.query:
        return True
    if any(
        message.role == item.source_role and item.value in message.content
        for message in request.history
    ):
        return True
    if summary is None or original_history is None:
        return False
    return any(
        fact.source_message_index < len(original_history)
        and original_history[fact.source_message_index].role == item.source_role
        and item.value in fact.content
        for fact in summary.facts
    )


def _planner_probe_signature(case, request, summary=None, original_history=None) -> tuple[str, ...]:
    required = [item for item in case.critical_context if item.planner_required]
    available = sorted(
        item.key for item in required
        if _item_is_available(item, request, summary, original_history)
    )
    if len(available) != len(required):
        return ("clarify", *available)
    return ("execute", case.case_id, *available)


def _aggregate_strategy(
    strategy: ContextStrategy,
    cases: list[ContextEvalCase],
    observations: list[ContextEvalObservation],
) -> ContextStrategyMetrics:
    selected = [item for item in observations if item.strategy == strategy]
    critical_total = sum(len(case.critical_context) for case in cases)
    protected_total = sum(
        item.kind in {"entity", "constraint"}
        for case in cases
        for item in case.critical_context
    )
    retained_total = sum(len(item.retained_keys) for item in selected)
    case_by_id = {case.case_id: case for case in cases}
    protected_lost = sum(
        critical.kind in {"entity", "constraint"} and critical.key in observation.lost_keys
        for observation in selected
        for critical in case_by_id[observation.case_id].critical_context
    )
    original_tokens = sum(item.metrics.original_tokens for item in selected)
    selected_tokens = sum(item.metrics.selected_tokens for item in selected)
    return ContextStrategyMetrics(
        strategy=strategy,
        case_count=len(selected),
        critical_context_retention_rate=retained_total / critical_total,
        history_token_compression_ratio=(selected_tokens / original_tokens if original_tokens else 1.0),
        planner_plan_equivalence_rate=(
            sum(item.planner_equivalent_to_full_history for item in selected) / len(selected)
        ),
        critical_entity_or_constraint_loss_rate=(
            protected_lost / protected_total if protected_total else 0.0
        ),
    )


def report_summary(report: ContextEvaluationReport) -> dict[str, object]:
    return {
        "case_count": report.case_count,
        "planner_probe": report.planner_probe,
        "last_n": report.last_n,
        "budget_tokens": report.budget_tokens,
        "summary_baseline_passed": report.summary_baseline_passed,
        "summary_failed_case_ids": report.summary_failed_case_ids,
        "strategies": [item.model_dump(mode="json") for item in report.strategies],
    }


def write_json_report(report: ContextEvaluationReport, path: Path) -> None:
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


class _OfflineSummaryProvider:
    """Deterministic extractive fixture; never used by production runtime."""

    def generate(self, messages, *, response_schema):
        del response_schema
        payload = json.loads(messages[1]["content"])
        facts = []
        for item in payload["summarized_history"]:
            if item["role"] != "user":
                continue
            content = item["content"]
            category = "planning_fact"
            if any(word in content for word in ("不要", "只", "仅", "必须", "更正", "改为", "为准", "先查")):
                category = "constraint"
            elif any(char.isdigit() for char in content):
                category = "entity"
            facts.append({
                "category": category,
                "content": content,
                "source_message_index": item["index"],
            })
        return {"facts": facts}


def _offline_summary_manager() -> ContextManager:
    from financial_agent.context.summarizer import HistorySummarizer
    return ContextManager(summarizer=HistorySummarizer(_OfflineSummaryProvider(), max_facts=100))
