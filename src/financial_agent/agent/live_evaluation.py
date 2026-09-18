"""Phase 6.4 live Qwen evaluation over deterministic local Tool fixtures."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from financial_agent.agent.control_plane_ablation import NoHistoryContextAdapter
from financial_agent.agent.e2e_evaluation import (
    ExpectedEvidence, LogicalTool, ToolInvocation, _fixture_registry, _logical_tools,
    _normalize_tool, _tool_key,
)
from financial_agent.agent.loop import AgentLoopResult, LoopPolicy
from financial_agent.agent.orchestrator import InvalidPlanError
from financial_agent.agent.retry import RetryPolicy
from financial_agent.answering.providers import (
    AnswerProviderResponseError, AnswerProviderTimeoutError, AnswerProviderUnavailableError,
)
from financial_agent.answering.service import AnswerWriter
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.context.summarizer import HistorySummarizer
from financial_agent.context.summary_providers import (
    SummaryProviderResponseError, SummaryProviderTimeoutError, SummaryProviderUnavailableError,
)
from financial_agent.observability import AgentTrace, InMemoryTraceSink, TraceSink, run_agent_loop_traced
from financial_agent.observability.models import (
    ComponentFailedEvent, ContextSelectedEvent, LoopIterationCompletedEvent, PlanProposedEvent,
    RetryScheduledEvent, RunFailedEvent, RunFinishedEvent, ToolAttemptEvent,
    VerifierCompletedEvent, WriterCompletedEvent,
)
from financial_agent.planner.providers import (
    PlannerProviderResponseError, PlannerProviderTimeoutError, PlannerProviderUnavailableError,
)
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.verifier.models import VerificationResult
from financial_agent.verifier.providers import (
    VerifierProviderResponseError, VerifierProviderTimeoutError, VerifierProviderUnavailableError,
)
from financial_agent.verifier.service import StructuredVerifier


LiveConfiguration = Literal["simple_baseline", "full_agent"]
FailureDomain = Literal[
    "none", "model_error", "agent_control_error", "tool/business_error",
    "provider/infra_error", "scorer/eval_error",
]
TraceHealth = Literal["healthy", "degraded"]
ProviderFactory = Callable[[str, str, LiveConfiguration, int], Any]


class LiveScenario(Schema):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)
    user_faults: list[Literal["timeout", "429", "503", "success"]] = Field(default_factory=list)
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    loop_policy: LoopPolicy = Field(default_factory=LoopPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    mechanisms: list[Literal[
        "single_tool", "multi_tool", "binding", "retry", "rewrite", "replan",
        "context", "clarify", "no_tool", "persistent_failure", "budget",
    ]] = Field(min_length=1)


class RequiredBinding(Schema):
    source_tool: str
    target_tool: str
    target_parameter: str
    source_path: list[str | int] = Field(min_length=1)


class LiveGold(Schema):
    case_id: str = Field(min_length=1)
    solvable: bool
    expected_decision: Literal["execute", "clarify", "no_tool"]
    acceptable_initial_plans: list[list[LogicalTool]] = Field(min_length=1)
    expected_tools: list[LogicalTool]
    required_text: list[str] = Field(default_factory=list)
    forbidden_text: list[str] = Field(default_factory=list)
    required_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    required_context_message_indexes: list[int] = Field(default_factory=list)
    required_binding: RequiredBinding | None = None
    expected_failure: bool = False


class FrozenLiveConfig(Schema):
    schema_version: Literal["1.0"]
    scenario_path: str
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_path: str
    gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=15, le=20)
    configurations: list[LiveConfiguration]
    execution_order: Literal["case_major_simple_then_full"]
    model: str = Field(min_length=1)
    temperature: Literal[0]
    max_infra_reruns: Literal[1]

    @model_validator(mode="after")
    def exact_configurations(self):
        if self.configurations != ["simple_baseline", "full_agent"]:
            raise ValueError("Live configurations must be fixed in baseline/full order")
        return self


class LiveEvalSet(Schema):
    manifest_sha256: str
    frozen_config: FrozenLiveConfig
    scenarios: list[LiveScenario]
    gold: list[LiveGold]


class FailureAttribution(Schema):
    domain: FailureDomain
    code: str
    component: str | None = None
    retryable_infra: bool = False


class LiveAudit(Schema):
    query: str
    gold: LiveGold
    initial_plan: dict[str, Any] | None = None
    replans: list[dict[str, Any]] = Field(default_factory=list)
    rewrites: list[dict[str, Any]] = Field(default_factory=list)
    tool_execution: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    final_evidence: list[dict[str, Any]] = Field(default_factory=list)
    failure: FailureAttribution


class LiveRunAttempt(Schema):
    run_attempt: int = Field(ge=1, le=2)
    trace_id: str
    valid_for_effect_metrics: bool
    task_success: bool
    answer_correct: bool
    planner_correct: bool
    tool_selection_correct: bool
    unnecessary_tool_calls: int = Field(ge=0)
    logical_tool_count: int = Field(ge=0)
    tool_attempts: int = Field(ge=0)
    loop_iterations: int = Field(ge=0)
    total_selected_history_tokens: int = Field(ge=0)
    selected_history_tokens_by_component: dict[str, int]
    latency_ms: float = Field(ge=0)
    trace_health: TraceHealth
    failure: FailureAttribution
    retry_triggered: bool
    rewrite_triggered: bool
    replan_triggered: bool
    context_requirement_met: bool
    result: AgentLoopResult | None = None
    audit: LiveAudit


class LiveCellResult(Schema):
    case_id: str
    configuration: LiveConfiguration
    attempts: list[LiveRunAttempt] = Field(min_length=1, max_length=2)
    selected_attempt: int = Field(ge=1, le=2)


class RecoveryMetric(Schema):
    opportunities: int = Field(ge=0)
    triggered: int = Field(ge=0)
    recovered: int = Field(ge=0)
    recovery_rate: float | None = Field(default=None, ge=0, le=1)
    all_triggered_runs: int = Field(default=0, ge=0)
    all_successful_triggered_runs: int = Field(default=0, ge=0)


class LiveAggregate(Schema):
    configuration: LiveConfiguration
    task_success_rate: float = Field(ge=0, le=1)
    final_answer_correctness: float = Field(ge=0, le=1)
    planner_correctness: float = Field(ge=0, le=1)
    recovery: dict[str, RecoveryMetric]
    total_tool_attempts: int = Field(ge=0)
    average_tool_attempts: float = Field(ge=0)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)
    average_loop_iterations: float = Field(ge=0)
    average_total_selected_history_tokens: float = Field(ge=0)
    average_selected_history_tokens_by_component: dict[str, float]
    latency_mean_ms: float = Field(ge=0)
    failure_types: dict[str, int]
    trace_health: dict[str, int]
    invalid_infra_attempts: int = Field(ge=0)


class LiveComparison(Schema):
    task_success_rate_delta: float
    final_answer_correctness_delta: float
    planner_correctness_delta: float
    average_tool_attempts_delta: float
    unnecessary_tool_call_rate_delta: float
    average_loop_iterations_delta: float
    average_total_selected_history_tokens_delta: float
    latency_mean_ms_delta: float


class LiveEvaluationReport(Schema):
    schema_version: Literal["1.0"] = "1.0"
    manifest_sha256: str
    frozen_config: FrozenLiveConfig
    case_count: int
    cell_count: int
    run_attempt_count: int
    aggregates: list[LiveAggregate]
    comparison: LiveComparison
    cells: list[LiveCellResult]
    harness_hard_gate_passed: bool
    hard_gate_failures: list[str]
    tuning_changes_after_freeze: list[str] = Field(default_factory=list)
    scorer_schema_fixes_after_freeze: list[str] = Field(default_factory=list)


class _PassThroughVerifier:
    def verify(self, request, plan, tool_results, draft):
        del request, plan, tool_results, draft
        return VerificationResult(decision="PASS", reason="Live baseline pass-through")


class _TeeSink:
    def __init__(self, memory: InMemoryTraceSink, external: TraceSink | None) -> None:
        self.memory, self.external = memory, external

    def write(self, trace: AgentTrace) -> None:
        self.memory.write(trace)
        if self.external is not None:
            self.external.write(trace)


def load_live_eval_set(config_path: Path) -> LiveEvalSet:
    raw = config_path.read_bytes()
    config = FrozenLiveConfig.model_validate_json(raw)
    root = config_path.parents[2].resolve()
    scenario_path = (root / config.scenario_path).resolve()
    gold_path = (root / config.gold_path).resolve()
    if root not in scenario_path.parents or root not in gold_path.parents:
        raise ValueError("Live eval source escapes repository root")
    if sha256(scenario_path.read_bytes()).hexdigest() != config.scenario_sha256:
        raise ValueError("Live scenario hash mismatch")
    if sha256(gold_path.read_bytes()).hexdigest() != config.gold_sha256:
        raise ValueError("Live gold hash mismatch")
    scenarios = _load_jsonl(scenario_path, LiveScenario)
    gold = _load_jsonl(gold_path, LiveGold)
    ids = [item.case_id for item in scenarios]
    if len(ids) != config.case_count or ids != [item.case_id for item in gold]:
        raise ValueError("Live case/gold IDs or count do not match frozen config")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate live case ID")
    required = {"single_tool", "multi_tool", "binding", "retry", "rewrite", "replan",
                "context", "clarify", "no_tool", "persistent_failure", "budget"}
    if not required.issubset({tag for item in scenarios for tag in item.mechanisms}):
        raise ValueError("Live holdout does not cover every required mechanism")
    return LiveEvalSet(
        manifest_sha256=sha256(raw).hexdigest(), frozen_config=config,
        scenarios=scenarios, gold=gold,
    )


def evaluate_live_agent(
    eval_set: LiveEvalSet, provider_factory: ProviderFactory, *, trace_sink: TraceSink | None = None,
) -> LiveEvaluationReport:
    cells: list[LiveCellResult] = []
    for scenario, gold in zip(eval_set.scenarios, eval_set.gold):
        for configuration in eval_set.frozen_config.configurations:
            attempts = [_run_attempt(scenario, gold, configuration, 1, provider_factory, trace_sink)]
            if attempts[0].failure.retryable_infra:
                attempts.append(_run_attempt(
                    scenario, gold, configuration, 2, provider_factory, trace_sink,
                ))
            selected = next(
                (index for index, item in reversed(list(enumerate(attempts, 1)))
                 if item.valid_for_effect_metrics), len(attempts),
            )
            cells.append(LiveCellResult(
                case_id=scenario.case_id, configuration=configuration,
                attempts=attempts, selected_attempt=selected,
            ))
    aggregates = [_aggregate(name, cells, eval_set.scenarios, eval_set.gold)
                  for name in eval_set.frozen_config.configurations]
    baseline, full = aggregates
    failures = []
    if any(attempt.trace_health == "degraded" for cell in cells for attempt in cell.attempts):
        failures.append("trace_health")
    if any(not cell.attempts for cell in cells):
        failures.append("missing_run_attempt")
    if any(attempt.failure.domain == "scorer/eval_error"
           for cell in cells for attempt in cell.attempts):
        failures.append("scorer_eval_error")
    return LiveEvaluationReport(
        manifest_sha256=eval_set.manifest_sha256, frozen_config=eval_set.frozen_config,
        case_count=len(eval_set.scenarios), cell_count=len(cells),
        run_attempt_count=sum(len(item.attempts) for item in cells), aggregates=aggregates,
        comparison=_comparison(baseline, full),
        cells=cells, harness_hard_gate_passed=not failures, hard_gate_failures=failures,
    )


def rescore_live_failure_attribution(
    report: LiveEvaluationReport, eval_set: LiveEvalSet,
) -> LiveEvaluationReport:
    """Apply the documented failure-classification correction without model calls."""
    cells = []
    for cell in report.cells:
        attempts = []
        for attempt in cell.attempts:
            failure = attempt.failure
            component = _component_for_code(failure.code) or failure.component
            if failure.code == "InvalidPlanError":
                failure = failure.model_copy(update={
                    "domain": "model_error", "component": "planner",
                })
            elif component != failure.component:
                failure = failure.model_copy(update={"component": component})
            audit = attempt.audit.model_copy(update={"failure": failure})
            attempts.append(attempt.model_copy(update={"failure": failure, "audit": audit}))
        cells.append(cell.model_copy(update={"attempts": attempts}))
    aggregates = [_aggregate(name, cells, eval_set.scenarios, eval_set.gold)
                  for name in eval_set.frozen_config.configurations]
    return report.model_copy(update={
        "cells": cells, "aggregates": aggregates,
        "comparison": _comparison(*aggregates),
        "scorer_schema_fixes_after_freeze": [
            "failure attribution v1.1: InvalidPlanError is model_error (invalid model plan), "
            "and typed provider errors identify planner/writer/verifier/summary components; "
            "rescored from persisted results and traces without Qwen calls"
        ],
    })


def _run_attempt(scenario, gold, configuration, run_attempt, provider_factory, external_sink):
    registry, call_context = _fixture_registry(scenario.user_faults)
    if configuration == "full_agent":
        summary = provider_factory("summary", scenario.case_id, configuration, run_attempt)
        context_manager = ContextManager(summarizer=HistorySummarizer(summary))
        retry_policy, loop_policy = scenario.retry_policy, scenario.loop_policy
        verifier = StructuredVerifier(
            provider_factory("verifier", scenario.case_id, configuration, run_attempt), registry,
            context_manager=context_manager, context_policy=scenario.context_policy,
        )
    else:
        context_manager = NoHistoryContextAdapter()
        retry_policy = scenario.retry_policy.model_copy(update={"max_retry": 0})
        loop_policy = scenario.loop_policy.model_copy(update={"max_rewrite": 0, "max_replan": 0})
        verifier = _PassThroughVerifier()
    planner = StructuredPlanner(
        provider_factory("planner", scenario.case_id, configuration, run_attempt), registry,
        context_manager=context_manager, context_policy=scenario.context_policy,
    )
    writer = AnswerWriter(
        provider_factory("writer", scenario.case_id, configuration, run_attempt), registry,
        context_manager=context_manager, context_policy=scenario.context_policy,
    )
    memory = InMemoryTraceSink()
    result = None
    exception: BaseException | None = None
    try:
        traced = run_agent_loop_traced(
            UserQuery(query=scenario.query, history=scenario.history), planner,
            PlanValidator(registry), registry, writer, verifier, policy=loop_policy,
            retry_policy=retry_policy, context=call_context, max_concurrency=1,
            sleeper=lambda _: None, capture_mode="evaluation",
            sink=_TeeSink(memory, external_sink),
        )
        result, trace = traced.result, traced.trace
    except BaseException as exc:
        exception = exc
        if not memory.traces:
            raise AssertionError("Exceptional live run did not produce a trace") from exc
        trace = memory.traces[-1]
    try:
        return _score_attempt(
            scenario, gold, configuration, run_attempt, registry, result, trace, exception,
        )
    except BaseException as scorer_error:
        return _scorer_failed_attempt(
            scenario, gold, run_attempt, result, trace, scorer_error,
        )


def _score_attempt(scenario, gold, configuration, run_attempt, registry, result, trace, exception):
    attempt_events = [item for item in trace.events if isinstance(item, ToolAttemptEvent)]
    invocations = [ToolInvocation(
        tool_name=item.tool_name, arguments=item.arguments.value,
        status=item.result_status, error_code=item.error_code,
    ) for item in attempt_events]
    actual_tools = _logical_tools(invocations)
    expected_tools = [_normalize_tool(registry, item) for item in gold.expected_tools]
    actual_counter = Counter(_tool_key(item) for item in actual_tools)
    expected_counter = Counter(_tool_key(item) for item in expected_tools)
    unnecessary = sum((actual_counter - expected_counter).values())
    plans = [item for item in trace.events if isinstance(item, PlanProposedEvent)]
    planner_correct = _planner_correct(plans[0] if plans else None, gold, registry)
    context_met = _context_requirement_met(trace, gold, planner_correct)
    answer_correct = _answer_correct_live(result, gold, registry, actual_tools)
    results_ok = bool(result and all(item.result.status != "error" for item in result.task_results))
    terminal_ok = bool(result and (
        (gold.expected_decision == "execute" and result.status == "completed")
        or result.status == gold.expected_decision
    ))
    task_success = bool(gold.solvable and terminal_ok and answer_correct and results_ok)
    failure = _attribute_failure(
        result, trace, exception, planner_correct, answer_correct, context_met, gold,
    )
    contexts = [item for item in trace.events if isinstance(item, ContextSelectedEvent)]
    breakdown = {component: 0 for component in ("planner", "writer", "verifier")}
    for event in contexts:
        breakdown[event.component] += event.metrics.selected_tokens
    terminal = [item for item in trace.events if isinstance(item, (RunFinishedEvent, RunFailedEvent))]
    latency = terminal[-1].elapsed_ms if terminal else max(
        (item.elapsed_ms for item in trace.events), default=0,
    )
    health = "degraded" if (
        trace.dropped_event_count or trace.recorder_error_codes
        or any(item.kind == "trace_degraded" for item in trace.events)
    ) else "healthy"
    writers = [item for item in trace.events if isinstance(item, WriterCompletedEvent)]
    audit = LiveAudit(
        query=scenario.query, gold=gold,
        initial_plan=plans[0].model_dump(mode="json") if plans else None,
        replans=[item.model_dump(mode="json") for item in plans[1:]],
        rewrites=[item.model_dump(mode="json") for item in writers if item.mode == "rewrite"],
        tool_execution=[item.model_dump(mode="json") for item in attempt_events],
        final_answer=result.answer if result else None,
        final_evidence=[item.model_dump(mode="json") for item in result.draft.evidence]
        if result and result.draft else [], failure=failure,
    )
    return LiveRunAttempt(
        run_attempt=run_attempt, trace_id=str(trace.trace_id),
        valid_for_effect_metrics=not failure.retryable_infra,
        task_success=task_success, answer_correct=answer_correct,
        planner_correct=planner_correct, tool_selection_correct=actual_counter == expected_counter,
        unnecessary_tool_calls=unnecessary, logical_tool_count=len(actual_tools),
        tool_attempts=len(attempt_events),
        loop_iterations=len([x for x in trace.events if isinstance(x, LoopIterationCompletedEvent)]),
        total_selected_history_tokens=sum(breakdown.values()),
        selected_history_tokens_by_component=breakdown, latency_ms=latency,
        trace_health=health, failure=failure,
        retry_triggered=any(isinstance(x, RetryScheduledEvent) for x in trace.events),
        rewrite_triggered=any(x.mode == "rewrite" for x in writers),
        replan_triggered=len(plans) > 1, context_requirement_met=context_met,
        result=result, audit=audit,
    )


def _planner_correct(event, gold, registry) -> bool:
    if event is None or event.decision != gold.expected_decision:
        return False
    if event.decision != "execute":
        return not event.tasks
    actual = Counter(_plan_key(task.tool_name, task.arguments.value, registry) for task in event.tasks)
    acceptable = any(actual == Counter(
        _plan_key(item.tool_name, item.arguments, registry) for item in plan
    ) for plan in gold.acceptable_initial_plans)
    if not acceptable:
        return False
    if gold.required_binding is None:
        return True
    tasks = {task.task_id: task for task in event.tasks if task.task_id is not None}
    binding = gold.required_binding
    return any(
        task.tool_name == binding.target_tool
        and any(
            item.target_parameter == binding.target_parameter
            and item.source_path == binding.source_path
            and tasks.get(item.source_task_id) is not None
            and tasks[item.source_task_id].tool_name == binding.source_tool
            for item in task.bindings
        )
        for task in event.tasks
    )


def _plan_key(tool_name, arguments, registry):
    arguments = dict(arguments)
    if any(value is None for value in arguments.values()):
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"
    try:
        return _tool_key(_normalize_tool(registry, LogicalTool(tool_name=tool_name, arguments=arguments)))
    except ValueError:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"


def _answer_correct_live(result, gold, registry, actual_tools):
    if gold.expected_decision in {"clarify", "no_tool"}:
        return bool(result and result.status == gold.expected_decision)
    if result is None or result.answer is None or result.draft is None:
        return False
    folded = result.answer.casefold()
    if any(text.casefold() not in folded for text in gold.required_text):
        return False
    if any(text.casefold() in folded for text in gold.forbidden_text):
        return False
    tasks = {task.task_id: task for task in result.plan}
    actual = set()
    for reference in result.draft.evidence:
        task = tasks.get(reference.task_id)
        if task is None:
            continue
        candidates = [item for item in actual_tools if item.tool_name == task.tool_name]
        actual.update((_tool_key(item), tuple(reference.source_path)) for item in candidates)
    required = {
        (_tool_key(_normalize_tool(registry, item)), tuple(item.source_path))
        for item in gold.required_evidence
    }
    return required.issubset(actual)


def _context_requirement_met(trace, gold, planner_correct):
    if not gold.required_context_message_indexes:
        return True
    observed = set()
    for event in trace.events:
        if isinstance(event, ContextSelectedEvent) and event.component == "planner":
            observed.update(event.selected_message_indexes)
            observed.update(event.retrieved_message_indexes)
    # A protected fact may arrive through a model summary rather than raw/retrieved
    # messages. In that case a correct initial plan is direct evidence that the
    # required history constraint survived selection.
    return set(gold.required_context_message_indexes).issubset(observed) or planner_correct


_INFRA_ERRORS = (
    PlannerProviderTimeoutError, PlannerProviderUnavailableError,
    AnswerProviderTimeoutError, AnswerProviderUnavailableError,
    VerifierProviderTimeoutError, VerifierProviderUnavailableError,
    SummaryProviderTimeoutError, SummaryProviderUnavailableError,
)
_MODEL_ERRORS = (
    PlannerProviderResponseError, AnswerProviderResponseError,
    VerifierProviderResponseError, SummaryProviderResponseError,
)


def _attribute_failure(result, trace, exception, planner_correct, answer_correct, context_met, gold):
    components = [x for x in trace.events if isinstance(x, ComponentFailedEvent)]
    component = _component_for_exception(exception)
    if component is None:
        component = components[-1].component if components else None
    if isinstance(exception, _INFRA_ERRORS):
        return FailureAttribution(domain="provider/infra_error", code=type(exception).__name__,
                                  component=component, retryable_infra=True)
    if isinstance(exception, _MODEL_ERRORS):
        return FailureAttribution(domain="model_error", code=type(exception).__name__, component=component)
    if isinstance(exception, InvalidPlanError):
        return FailureAttribution(domain="model_error", code="InvalidPlanError", component="planner")
    if exception is not None:
        return FailureAttribution(domain="agent_control_error", code=type(exception).__name__,
                                  component=component)
    if result is not None and result.stop_reason in {
        "total_tool_budget", "max_iterations", "max_rewrite", "max_replan", "no_progress",
    }:
        return FailureAttribution(domain="agent_control_error", code=result.stop_reason)
    tool_errors = [x for x in trace.events
                   if isinstance(x, ToolAttemptEvent) and x.result_status == "error"]
    if tool_errors and not answer_correct:
        return FailureAttribution(domain="tool/business_error",
                                  code=tool_errors[-1].error_code or "tool_error")
    if not context_met:
        return FailureAttribution(domain="agent_control_error", code="context_information_missed")
    if not planner_correct:
        return FailureAttribution(domain="model_error", code="planner_incorrect")
    if gold.expected_failure:
        return FailureAttribution(domain="none", code="expected_failure_observed")
    if not answer_correct:
        return FailureAttribution(domain="model_error", code="answer_incorrect")
    return FailureAttribution(domain="none", code="success")


def _scorer_failed_attempt(scenario, gold, run_attempt, result, trace, error):
    failure = FailureAttribution(
        domain="scorer/eval_error", code=type(error).__name__, component="scorer",
    )
    terminal = [item for item in trace.events if isinstance(item, (RunFinishedEvent, RunFailedEvent))]
    latency = terminal[-1].elapsed_ms if terminal else max(
        (item.elapsed_ms for item in trace.events), default=0,
    )
    return LiveRunAttempt(
        run_attempt=run_attempt, trace_id=str(trace.trace_id), valid_for_effect_metrics=True,
        task_success=False, answer_correct=False, planner_correct=False,
        tool_selection_correct=False, unnecessary_tool_calls=0, logical_tool_count=0,
        tool_attempts=len([x for x in trace.events if isinstance(x, ToolAttemptEvent)]),
        loop_iterations=len([x for x in trace.events if isinstance(x, LoopIterationCompletedEvent)]),
        total_selected_history_tokens=0,
        selected_history_tokens_by_component={"planner": 0, "writer": 0, "verifier": 0},
        latency_ms=latency, trace_health="healthy", failure=failure,
        retry_triggered=False, rewrite_triggered=False, replan_triggered=False,
        context_requirement_met=False, result=result,
        audit=LiveAudit(query=scenario.query, gold=gold,
                        final_answer=result.answer if result else None, failure=failure),
    )


def _component_for_exception(error):
    return _component_for_code(type(error).__name__) if error is not None else None


def _component_for_code(code):
    if code.startswith("Planner") or code == "InvalidPlanError":
        return "planner"
    if code.startswith("Answer"):
        return "writer"
    if code.startswith("Verifier"):
        return "verifier"
    if code.startswith("Summary"):
        return "summary"
    return None


def _aggregate(configuration, cells, scenarios, gold):
    by_case = {item.case_id: item for item in scenarios}
    solvable_by_case = {item.case_id: item.solvable for item in gold}
    selected_cells = [item for item in cells if item.configuration == configuration]
    selected = [item.attempts[item.selected_attempt - 1] for item in selected_cells]
    solvable = [item for item, cell in zip(selected, selected_cells)
                if solvable_by_case[cell.case_id]]
    recovery = {}
    for mechanism, attr in (("retry", "retry_triggered"), ("rewrite", "rewrite_triggered"),
                            ("replan", "replan_triggered"), ("context", "context_requirement_met")):
        pairs = [(by_case[cell.case_id], item) for cell, item in zip(selected_cells, selected)
                 if mechanism in by_case[cell.case_id].mechanisms]
        triggered = [item for _, item in pairs if getattr(item, attr)]
        recovered = sum(item.task_success for item in triggered)
        all_triggered = triggered if mechanism == "context" else [
            item for item in selected if getattr(item, attr)
        ]
        recovery[mechanism] = RecoveryMetric(
            opportunities=len(pairs), triggered=len(triggered), recovered=recovered,
            recovery_rate=recovered / len(triggered) if triggered else None,
            all_triggered_runs=len(all_triggered),
            all_successful_triggered_runs=sum(item.task_success for item in all_triggered),
        )
    count = len(selected)
    logical = sum(item.logical_tool_count for item in selected)
    return LiveAggregate(
        configuration=configuration,
        task_success_rate=_rate(sum(x.task_success for x in solvable), len(solvable)),
        final_answer_correctness=_rate(sum(x.answer_correct for x in solvable), len(solvable)),
        planner_correctness=_rate(sum(x.planner_correct for x in selected), count), recovery=recovery,
        total_tool_attempts=sum(x.tool_attempts for x in selected),
        average_tool_attempts=sum(x.tool_attempts for x in selected) / count,
        unnecessary_tool_call_rate=_rate(sum(x.unnecessary_tool_calls for x in selected), logical),
        average_loop_iterations=sum(x.loop_iterations for x in selected) / count,
        average_total_selected_history_tokens=sum(x.total_selected_history_tokens for x in selected) / count,
        average_selected_history_tokens_by_component={
            component: sum(x.selected_history_tokens_by_component[component] for x in selected) / count
            for component in ("planner", "writer", "verifier")
        },
        latency_mean_ms=sum(x.latency_ms for x in selected) / count,
        failure_types=dict(sorted(Counter(f"{x.failure.domain}:{x.failure.code}" for x in selected).items())),
        trace_health=dict(sorted(Counter(x.trace_health for x in selected).items())),
        invalid_infra_attempts=sum(
            attempt.failure.retryable_infra for cell in selected_cells for attempt in cell.attempts
        ),
    )


def _load_jsonl(path, model):
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                values.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid live eval entry at {path}:{line_number}") from exc
    if not values:
        raise ValueError(f"Live eval file is empty: {path}")
    return values


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 1.0


def _comparison(baseline, full):
    return LiveComparison(
        task_success_rate_delta=full.task_success_rate - baseline.task_success_rate,
        final_answer_correctness_delta=(full.final_answer_correctness - baseline.final_answer_correctness),
        planner_correctness_delta=full.planner_correctness - baseline.planner_correctness,
        average_tool_attempts_delta=full.average_tool_attempts - baseline.average_tool_attempts,
        unnecessary_tool_call_rate_delta=(full.unnecessary_tool_call_rate - baseline.unnecessary_tool_call_rate),
        average_loop_iterations_delta=full.average_loop_iterations - baseline.average_loop_iterations,
        average_total_selected_history_tokens_delta=(
            full.average_total_selected_history_tokens - baseline.average_total_selected_history_tokens
        ),
        latency_mean_ms_delta=full.latency_mean_ms - baseline.latency_mean_ms,
    )
