"""Holdout ablation across complete Context Manager strategies."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from financial_agent.agent.loop import LoopPolicy, run_agent_loop
from financial_agent.agent.retry import RetryPolicy
from financial_agent.answering.service import AnswerWriter
from financial_agent.context.manager import ContextManager
from financial_agent.context.models import ContextPolicy, ContextSelection
from financial_agent.context.summarizer import HistorySummarizer
from financial_agent.context.summary_prompt import build_summary_messages
from financial_agent.context.token_estimation import HeuristicTokenEstimator
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.contracts import ToolResult
from financial_agent.verifier.service import StructuredVerifier


STRATEGIES = (
    "full_history", "last_n", "budgeted_selection", "summary_compression", "summary_retrieval",
)


class AblationCriticalFact(Schema):
    value: str = Field(min_length=1)
    protected: bool = False


class AblationCase(Schema):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[Message] = Field(min_length=1)
    checkpoint_turns: list[int] = Field(default_factory=list)
    critical_context: list[AblationCriticalFact] = Field(min_length=1)
    expected_arguments: dict[str, Any]
    answer_value: str = Field(min_length=1)


class AblationRun(Schema):
    case_id: str
    strategy: str
    task_correct: bool
    planner_correct: bool
    plan_equivalent_to_full: bool = False
    critical_retention: float = Field(ge=0, le=1)
    protected_retention: float = Field(ge=0, le=1)
    original_tokens: int = Field(ge=0)
    selected_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0, le=1)
    context_latency_ms: float = Field(ge=0)
    end_to_end_latency_ms: float = Field(ge=0)
    summary_provider_call_count: int = Field(ge=0)
    summary_rebuild_count: int = Field(ge=0)
    summary_incremental_update_count: int = Field(ge=0)
    summary_input_tokens: int = Field(ge=0)
    repeated_rebuild_counterfactual_tokens: int = Field(ge=0)
    summary_prefix_stability_ratio: float | None = Field(default=None, ge=0, le=1)
    retrieval_active: bool
    retrieved_turn_count: int = Field(ge=0)
    failure: str | None = None


class AblationStrategyMetrics(Schema):
    strategy: str
    case_count: int
    task_accuracy: float
    planner_accuracy: float
    plan_equivalence: float
    critical_retention: float
    protected_retention: float
    history_tokens: int
    compression_ratio: float
    context_latency_mean_ms: float
    context_latency_median_ms: float
    end_to_end_latency_mean_ms: float
    end_to_end_latency_median_ms: float
    summary_provider_call_count: int
    summary_rebuild_count: int
    summary_incremental_update_count: int
    summary_input_tokens: int
    repeated_rebuild_counterfactual_tokens: int
    summary_prefix_stability_ratio: float | None = None
    retrieval_active_rate: float
    retrieved_turn_count: int


class AblationReport(Schema):
    mode: Literal["offline", "live"]
    holdout_sha256: str
    budget_tokens: int
    last_n: int
    case_count: int
    strategies: list[AblationStrategyMetrics]
    runs: list[AblationRun]
    hard_gate_passed: bool
    hard_gate_failures: list[str] = Field(default_factory=list)
    experimental_targets: dict[str, bool | float | int]


class _LookupInput(Schema):
    topic: str
    entity: str
    start_date: str | None = None
    end_date: str | None = None
    constraint: str | None = None


class _LookupOutput(Schema):
    value: str


class _Registry:
    def describe(self):
        return [{
            "name": "lookup_financial_fact",
            "description": (
                "Look up the requested synthetic financial fact. Copy topic, entity, dates and explicit constraint "
                "from the current query or conversation context."
            ),
            "input_schema": _LookupInput.model_json_schema(),
        }]

    def input_model(self, name):
        return _LookupInput if name == "lookup_financial_fact" else None

    def output_model(self, name):
        return _LookupOutput if name == "lookup_financial_fact" else None

    def invoke(self, name, arguments, *, context, request_id=None):
        del context
        value = f"{arguments['topic']}:{arguments['entity']}"
        return ToolResult(
            status="success", data={"value": value}, source="synthetic_ablation",
            latency=0, error=None, request_id=request_id or uuid4(),
        )


class _RecordingContextManager:
    def __init__(self, delegate: ContextManager) -> None:
        self.delegate = delegate
        self.selections: list[ContextSelection] = []
        self.latencies: list[float] = []
        self.calls: list[tuple[UserQuery, ContextPolicy, ContextSelection]] = []

    def select(self, request, component, policy):
        started = perf_counter()
        selection = self.delegate.select(request, component, policy)
        self.latencies.append((perf_counter() - started) * 1_000)
        self.selections.append(selection)
        self.calls.append((request, policy, selection))
        return selection


class _OfflinePlannerProvider:
    def __init__(self, case: AblationCase) -> None:
        self.case = case

    def generate(self, messages, *, response_schema):
        del response_schema
        visible = json.dumps(messages, ensure_ascii=False)
        if not all(item.value in visible for item in self.case.critical_context):
            return {"decision": "clarify", "tasks": []}
        return {
            "decision": "execute",
            "tasks": [{
                "task_id": "t1", "tool_name": "lookup_financial_fact",
                "arguments": self.case.expected_arguments, "dependencies": [], "bindings": [],
            }],
        }


class _OfflineAnswerProvider:
    def generate(self, messages, *, response_schema):
        del response_schema
        payload = json.loads(messages[-1]["content"])
        value = payload["tool_results"][0]["data"]["value"]
        return {
            "answer": value,
            "evidence": [{"task_id": "t1", "source_path": ["value"]}],
        }


class _OfflineVerifierProvider:
    def generate(self, messages, *, response_schema):
        del messages, response_schema
        return {"decision": "PASS", "reason": "all expected evidence is present", "missing_evidence": []}


class OfflineAblationSummaryProvider:
    """Stable deterministic summary fixture that intentionally omits assistant planning facts."""

    def generate(self, messages, *, response_schema):
        del response_schema
        payload = json.loads(messages[1]["content"])
        source = payload.get("summarized_history") or payload.get("newly_aged_out_history", [])
        facts = []
        for item in source:
            if item["role"] != "user" or item["content"] in {"好的", "闲聊"}:
                continue
            content = item["content"]
            if any(word in content for word in ("不要", "不得", "必须", "更正", "纠正", "改为", "为准", "取消")):
                category = "constraint"
            elif "syn-user-" in content or any(char.isdigit() for char in content):
                category = "entity"
            else:
                category = "planning_fact"
            facts.append({
                "category": category, "content": content, "source_message_index": item["index"],
            })
        if "existing_summary" in payload:
            return {"additions": facts, "replacements": []}
        return {"facts": facts}


def load_ablation_cases(path: Path) -> list[AblationCase]:
    cases = [
        AblationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(cases) != 12:
        raise ValueError("Phase 5.4 holdout must contain exactly 12 cases")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Ablation case IDs must be unique")
    return cases


def evaluate_context_ablation(
    cases: list[AblationCase],
    *,
    mode: Literal["offline", "live"] = "offline",
    budget_tokens: int = 160,
    last_n: int = 3,
    live_factories: tuple[Any, Any, Any, Any] | None = None,
    holdout_sha256: str,
) -> AblationReport:
    runs: list[AblationRun] = []
    for case in cases:
        case_runs = [
            _run_case(case, strategy, mode, budget_tokens, last_n, live_factories)
            for strategy in STRATEGIES
        ]
        full_signature = _plan_signature(case_runs[0])
        for run in case_runs:
            run.plan_equivalent_to_full = _plan_signature(run) == full_signature
        runs.extend(case_runs)
    strategies = [_aggregate(strategy, runs) for strategy in STRATEGIES]
    hard_failures = [
        f"{run.case_id}:{run.strategy}:{run.failure or 'protected_or_budget_failure'}"
        for run in runs
        if run.failure is not None
        or (
            run.strategy in {"summary_compression", "summary_retrieval"}
            and run.protected_retention < 1
        )
        or (
            run.strategy in {"budgeted_selection", "summary_compression", "summary_retrieval"}
            and run.selected_tokens > budget_tokens
        )
    ]
    by_strategy = {item.strategy: item for item in strategies}
    retrieval = by_strategy["summary_retrieval"]
    last = by_strategy["last_n"]
    full = by_strategy["full_history"]
    targets = {
        "history_token_ratio_le_0_60": retrieval.compression_ratio <= 0.60,
        "summary_prefix_stability_ge_0_70": (
            retrieval.summary_prefix_stability_ratio is not None
            and retrieval.summary_prefix_stability_ratio >= 0.70
        ),
        "task_accuracy_not_below_last_n": retrieval.task_accuracy >= last.task_accuracy,
        "planner_accuracy_not_below_last_n": retrieval.planner_accuracy >= last.planner_accuracy,
        "task_accuracy_within_one_case_of_full": (
            full.task_accuracy - retrieval.task_accuracy <= 1 / len(cases)
        ),
        "planner_accuracy_within_one_case_of_full": (
            full.planner_accuracy - retrieval.planner_accuracy <= 1 / len(cases)
        ),
        "incremental_input_below_rebuild_counterfactual": (
            retrieval.summary_input_tokens < retrieval.repeated_rebuild_counterfactual_tokens
        ),
    }
    return AblationReport(
        mode=mode, holdout_sha256=holdout_sha256, budget_tokens=budget_tokens, last_n=last_n,
        case_count=len(cases), strategies=strategies, runs=runs,
        hard_gate_passed=not hard_failures, hard_gate_failures=hard_failures,
        experimental_targets=targets,
    )


def _run_case(case, strategy, mode, budget, last_n, live_factories) -> AblationRun:
    summary_provider = (
        OfflineAblationSummaryProvider() if mode == "offline" else live_factories[0]()
    )
    manager = ContextManager(summarizer=HistorySummarizer(summary_provider, max_facts=100))
    recorder = _RecordingContextManager(manager)
    policy = ContextPolicy(
        strategy=strategy, budget_tokens=budget, last_n=last_n, summary_recent_n=2,
    )
    turns = _turns(case.history)
    for count in case.checkpoint_turns:
        recorder.select(
            UserQuery(query="继续", history=[message for turn in turns[:count] for message in turn]),
            "planner", policy,
        )
    before_final = len(recorder.selections)
    registry = _Registry()
    if mode == "offline":
        planner_provider = _OfflinePlannerProvider(case)
        answer_provider = _OfflineAnswerProvider()
        verifier_provider = _OfflineVerifierProvider()
    else:
        planner_provider, answer_provider, verifier_provider = (
            live_factories[1](), live_factories[2](), live_factories[3]()
        )
    planner = StructuredPlanner(planner_provider, registry, context_manager=recorder, context_policy=policy)
    writer = AnswerWriter(answer_provider, registry, context_manager=recorder, context_policy=policy)
    verifier = StructuredVerifier(verifier_provider, registry, context_manager=recorder, context_policy=policy)
    failure = None
    started = perf_counter()
    try:
        result = run_agent_loop(
            UserQuery(query=case.query, history=case.history), planner, PlanValidator(registry),
            registry, writer, verifier, policy=LoopPolicy(max_rewrite=0, max_replan=0),
            retry_policy=RetryPolicy(max_retry=0), sleeper=lambda _: None,
        )
        plan = result.plan
        planner_correct = (
            result.status == "completed" and len(plan) == 1
            and plan[0].tool_name == "lookup_financial_fact"
            and plan[0].arguments == case.expected_arguments
        )
        task_correct = planner_correct and result.answer is not None and case.answer_value in result.answer
    except Exception as exc:  # Per-case report retains provider/schema failures without leaking content.
        result = None
        plan = []
        planner_correct = task_correct = False
        failure = type(exc).__name__
    elapsed = (perf_counter() - started) * 1_000
    final_selections = recorder.selections[before_final:]
    final = final_selections[0] if final_selections else recorder.selections[-1]
    visible = _selection_text(final)
    retained = [item for item in case.critical_context if item.value in visible]
    protected = [item for item in case.critical_context if item.protected]
    metrics = [selection.metrics for selection in recorder.selections]
    ratios = [
        item.summary_prefix_stability_ratio for item in metrics
        if item.summary_prefix_stability_ratio is not None
        and item.summary_incremental_update_count > 0
    ]
    run = AblationRun(
        case_id=case.case_id, strategy=strategy, task_correct=task_correct,
        planner_correct=planner_correct,
        critical_retention=len(retained) / len(case.critical_context),
        protected_retention=(
            sum(item in retained for item in protected) / len(protected) if protected else 1.0
        ),
        original_tokens=final.metrics.original_tokens,
        selected_tokens=final.metrics.selected_tokens,
        compression_ratio=final.metrics.compression_ratio,
        context_latency_ms=sum(recorder.latencies), end_to_end_latency_ms=elapsed,
        summary_provider_call_count=sum(item.summary_provider_call_count for item in metrics),
        summary_rebuild_count=sum(item.summary_rebuild_count for item in metrics),
        summary_incremental_update_count=sum(item.summary_incremental_update_count for item in metrics),
        summary_input_tokens=sum(item.summary_input_tokens for item in metrics),
        repeated_rebuild_counterfactual_tokens=_rebuild_counterfactual_tokens(recorder.calls),
        summary_prefix_stability_ratio=mean(ratios) if ratios else None,
        retrieval_active=any(item.retrieval_active for item in (s.metrics for s in final_selections)),
        retrieved_turn_count=sum(item.retrieved_turn_count for item in (s.metrics for s in final_selections)),
        failure=failure,
    )
    run.__dict__["_plan"] = plan
    return run


def _plan_signature(run: AblationRun):
    plan = run.__dict__.get("_plan", [])
    return tuple(
        (task.tool_name, json.dumps(task.arguments, ensure_ascii=False, sort_keys=True), tuple(task.dependencies))
        for task in plan
    )


def _selection_text(selection: ContextSelection) -> str:
    payload = {
        "query": selection.request.query,
        "history": [message.model_dump(mode="json") for message in selection.request.history],
        "summary": selection.summary.model_dump(mode="json") if selection.summary else None,
        "retrieved": [item.model_dump(mode="json") for item in selection.retrieved_history],
    }
    return json.dumps(payload, ensure_ascii=False)


def _turns(history):
    turns = []
    current = []
    for message in history:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _rebuild_counterfactual_tokens(
    calls: list[tuple[UserQuery, ContextPolicy, ContextSelection]],
) -> int:
    """Estimate the same updates if every changed prefix used a full rebuild prompt."""
    total = 0
    estimator = HeuristicTokenEstimator()
    for request, policy, selection in calls:
        if selection.metrics.summary_provider_call_count == 0:
            continue
        summarized = _summarized_prefix(request.history, policy, estimator)
        if summarized:
            total += _message_payload_tokens(build_summary_messages(summarized))
    return total


def _summarized_prefix(
    history: list[Message],
    policy: ContextPolicy,
    estimator: HeuristicTokenEstimator,
) -> list[tuple[int, Message]]:
    indexed_turns: list[list[tuple[int, Message]]] = []
    current: list[tuple[int, Message]] = []
    for index, message in enumerate(history):
        if message.role == "user" and current:
            indexed_turns.append(current)
            current = []
        current.append((index, message))
    if current:
        indexed_turns.append(current)
    recent_count = min(policy.summary_recent_n, len(indexed_turns))
    prefix = list(indexed_turns[:-recent_count]) if recent_count else list(indexed_turns)
    recent = list(indexed_turns[-recent_count:]) if recent_count else []
    if policy.strategy == "summary_compression":
        raw_target = int(policy.budget_tokens * (1 - policy.summary_budget_ratio))
        while recent and estimator.estimate_messages(
            [message for turn in recent for _, message in turn]
        ) > raw_target:
            prefix.append(recent.pop(0))
    return [item for turn in prefix for item in turn]


def _message_payload_tokens(messages: list[dict[str, str]]) -> int:
    compact = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return (len(compact.encode("utf-8")) + 3) // 4


def _aggregate(strategy: str, runs: list[AblationRun]) -> AblationStrategyMetrics:
    selected = [item for item in runs if item.strategy == strategy]
    ratios = [item.summary_prefix_stability_ratio for item in selected if item.summary_prefix_stability_ratio is not None]
    original = sum(item.original_tokens for item in selected)
    chosen = sum(item.selected_tokens for item in selected)
    return AblationStrategyMetrics(
        strategy=strategy, case_count=len(selected),
        task_accuracy=mean(item.task_correct for item in selected),
        planner_accuracy=mean(item.planner_correct for item in selected),
        plan_equivalence=mean(item.plan_equivalent_to_full for item in selected),
        critical_retention=mean(item.critical_retention for item in selected),
        protected_retention=mean(item.protected_retention for item in selected),
        history_tokens=chosen, compression_ratio=chosen / original if original else 1,
        context_latency_mean_ms=mean(item.context_latency_ms for item in selected),
        context_latency_median_ms=median(item.context_latency_ms for item in selected),
        end_to_end_latency_mean_ms=mean(item.end_to_end_latency_ms for item in selected),
        end_to_end_latency_median_ms=median(item.end_to_end_latency_ms for item in selected),
        summary_provider_call_count=sum(item.summary_provider_call_count for item in selected),
        summary_rebuild_count=sum(item.summary_rebuild_count for item in selected),
        summary_incremental_update_count=sum(item.summary_incremental_update_count for item in selected),
        summary_input_tokens=sum(item.summary_input_tokens for item in selected),
        repeated_rebuild_counterfactual_tokens=sum(item.repeated_rebuild_counterfactual_tokens for item in selected),
        summary_prefix_stability_ratio=mean(ratios) if ratios else None,
        retrieval_active_rate=mean(item.retrieval_active for item in selected),
        retrieved_turn_count=sum(item.retrieved_turn_count for item in selected),
    )
