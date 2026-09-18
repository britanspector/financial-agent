"""Offline Phase 6.3 control-plane ablation over complete Agent runs."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from financial_agent.agent.e2e_evaluation import (
    E2EExpectation, E2EScenario, LogicalTool, ReplayCall, ToolInvocation,
    _answer_correct, _fixture_registry, _logical_tools, _normalize_tool, _tool_key,
    load_e2e_eval_set,
)
from financial_agent.agent.loop import AgentLoopResult
from financial_agent.answering.service import AnswerWriter
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.observability import AgentTrace, InMemoryTraceSink, TraceSink, run_agent_loop_traced
from financial_agent.observability.models import (
    ContextSelectedEvent, LoopIterationCompletedEvent, PlanProposedEvent,
    RetryScheduledEvent, RunFailedEvent, RunFinishedEvent, ToolAttemptBlockedEvent,
    ToolAttemptEvent, VerifierCompletedEvent, WriterCompletedEvent,
)
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Schema, UserQuery
from financial_agent.verifier.models import VerificationResult
from financial_agent.verifier.service import StructuredVerifier


AblationConfigName = Literal[
    "simple_baseline", "retry", "verifier_rewrite", "replan", "full_agent",
]
Mechanism = Literal["control", "expected_failure", "retry", "rewrite", "replan", "context"]
TraceHealth = Literal["healthy", "degraded"]
RunFailureType = Literal[
    "component_exception", "deadline_exhausted", "tool_budget_exhausted",
    "max_iterations", "max_rewrite", "max_replan", "no_progress",
    "retryable_tool_error", "non_retryable_tool_error", "clarify", "no_tool",
    "task_result_error", "incorrect_answer", "success", "unclassified",
]


class AblationConfiguration(Schema):
    name: AblationConfigName
    retry_enabled: bool
    verifier_rewrite_enabled: bool
    replan_enabled: bool
    history_context_enabled: bool


ABLATION_CONFIGURATIONS = [
    AblationConfiguration(name="simple_baseline", retry_enabled=False,
                          verifier_rewrite_enabled=False, replan_enabled=False,
                          history_context_enabled=False),
    AblationConfiguration(name="retry", retry_enabled=True,
                          verifier_rewrite_enabled=False, replan_enabled=False,
                          history_context_enabled=False),
    AblationConfiguration(name="verifier_rewrite", retry_enabled=True,
                          verifier_rewrite_enabled=True, replan_enabled=False,
                          history_context_enabled=False),
    AblationConfiguration(name="replan", retry_enabled=True,
                          verifier_rewrite_enabled=True, replan_enabled=True,
                          history_context_enabled=False),
    AblationConfiguration(name="full_agent", retry_enabled=True,
                          verifier_rewrite_enabled=True, replan_enabled=True,
                          history_context_enabled=True),
]


class ManifestSource(Schema):
    role: Literal[
        "phase61_scenarios", "phase61_expectations", "phase63_scenarios",
        "phase63_expectations", "counterfactuals",
    ]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestCase(Schema):
    case_id: str = Field(min_length=1)
    mechanism: Mechanism


class CombinedManifest(Schema):
    schema_version: Literal["1.0"]
    sources: list[ManifestSource]
    cases: list[ManifestCase]

    @model_validator(mode="after")
    def unique_entries(self):
        if len({item.role for item in self.sources}) != len(self.sources):
            raise ValueError("Combined manifest source roles must be unique")
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("Combined manifest case IDs must be unique")
        required = {"retry", "rewrite", "replan", "context"}
        actual = {item.mechanism for item in self.cases}
        if not required.issubset(actual):
            raise ValueError("Combined manifest must contain every mechanism case")
        return self


class CounterfactualRule(Schema):
    case_id: str
    component: Literal["planner", "writer", "verifier"]
    trigger: Literal["missing_required_prompt", "tool_error"]
    call_index: int = Field(default=0, ge=0)
    output: dict[str, Any]

    @model_validator(mode="after")
    def valid_trigger_component(self):
        allowed = {
            "missing_required_prompt": {"planner"},
            "tool_error": {"writer"},
        }
        if self.component not in allowed[self.trigger]:
            raise ValueError("Counterfactual trigger is invalid for this component")
        return self


class AblationEvalSet(Schema):
    manifest_hash: str
    manifest: CombinedManifest
    scenarios: list[E2EScenario]
    expectations: list[E2EExpectation]
    counterfactuals: list[CounterfactualRule]


class TraceEventPointer(Schema):
    kind: Literal[
        "context_selected", "tool_attempt", "retry_scheduled", "writer_completed",
        "verifier_completed", "plan_proposed",
    ]
    sequence: int = Field(ge=1)
    task_ref: str | None = None


class MechanismAttribution(Schema):
    case_id: str
    mechanism: Literal["retry", "rewrite", "replan", "context"]
    before_config: AblationConfigName
    after_config: AblationConfigName
    before_trace_id: str
    after_trace_id: str
    before_failure_type: RunFailureType
    after_failure_type: RunFailureType
    status: Literal["credited", "not_recovered", "already_solved", "regression"]
    evidence: list[TraceEventPointer] = Field(default_factory=list)


class AblationCellResult(Schema):
    case_id: str
    configuration: AblationConfigName
    mechanism: Mechanism
    solvable: bool
    task_success: bool
    answer_correct: bool
    tool_selection_correct: bool
    unnecessary_tool_calls: int = Field(ge=0)
    logical_tool_count: int = Field(ge=0)
    tool_attempts: int = Field(ge=0)
    loop_iterations: int = Field(ge=0)
    total_selected_history_tokens: int = Field(ge=0)
    selected_history_tokens_by_component: dict[Literal["planner", "writer", "verifier"], int]
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    run_failure_type: RunFailureType
    trace_health: TraceHealth
    exception_type: str | None = None
    result: AgentLoopResult | None = None
    trace: AgentTrace


class AblationMetrics(Schema):
    configuration: AblationConfigName
    task_success_rate: float = Field(ge=0, le=1)
    final_answer_correctness: float = Field(ge=0, le=1)
    recovery_rate: float = Field(ge=0, le=1)
    total_tool_attempts: int = Field(ge=0)
    average_tool_attempts: float = Field(ge=0)
    total_unnecessary_tool_calls: int = Field(ge=0)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)
    average_loop_iterations: float = Field(ge=0)
    average_total_selected_history_tokens: float = Field(ge=0)
    average_selected_history_tokens_by_component: dict[
        Literal["planner", "writer", "verifier"], float
    ]
    latency_mean_ms: float = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    latency_is_small_sample_descriptive: Literal[True] = True
    run_failure_types: dict[str, int]
    trace_health: dict[str, int]


class AblationDelta(Schema):
    before_config: AblationConfigName
    after_config: AblationConfigName
    task_success_rate_delta: float
    final_answer_correctness_delta: float
    recovery_rate_delta: float
    average_tool_attempts_delta: float
    unnecessary_tool_call_rate_delta: float
    average_loop_iterations_delta: float
    average_total_selected_history_tokens_delta: float
    latency_mean_ms_delta: float


class ControlPlaneAblationReport(Schema):
    manifest_hash: str
    case_count: int = Field(ge=1)
    configuration_count: Literal[5] = 5
    run_count: int = Field(ge=1)
    metrics: list[AblationMetrics]
    deltas: list[AblationDelta]
    attributions: list[MechanismAttribution]
    cells: list[AblationCellResult]
    phase61_full_compatible: bool
    recovery_staircase_valid: bool
    all_passed: bool


class NoHistoryContextAdapter:
    """Eval-only explicit context-off behavior: current query, no history."""

    def __init__(self) -> None:
        self._manager = ContextManager()
        self._empty_policy = ContextPolicy(strategy="full_history")

    def select(self, request: UserQuery, component, policy):
        del policy
        stripped = request.model_copy(update={"history": []})
        return self._manager.select(stripped, component, self._empty_policy)


class _PassThroughVerifier:
    def verify(self, request, plan, tool_results, draft):
        del request, plan, tool_results, draft
        return VerificationResult(decision="PASS", reason="Ablation pass-through verifier")


class _AblationReplayProvider:
    def __init__(self, case_id: str, component: str, calls: list[ReplayCall],
                 rules: list[CounterfactualRule]) -> None:
        self._case_id = case_id
        self._component = component
        self._calls = list(calls)
        self._index = 0
        self._rules = {(item.component, item.trigger, item.call_index): item for item in rules}

    def generate(self, messages, *, response_schema):
        del response_schema
        if self._index >= len(self._calls):
            raise AssertionError("Unexpected ablation replay provider call")
        call = self._calls[self._index]
        prompt_content = "\n".join(str(item.get("content", "")) for item in messages)
        missing = [item for item in call.required_prompt_fragments if item not in prompt_content]
        trigger = None
        if missing:
            trigger = "missing_required_prompt"
        elif self._component == "writer" and (
            '"status": "error"' in prompt_content or '"status":"error"' in prompt_content
        ):
            trigger = "tool_error"
        rule = self._rules.get((self._component, trigger, self._index)) if trigger else None
        self._index += 1
        if rule is not None:
            return rule.output
        if missing:
            raise AssertionError(f"Required prompt fragments missing: {missing}")
        return call.output


class _TeeSink:
    def __init__(self, memory: InMemoryTraceSink, external: TraceSink | None) -> None:
        self._memory = memory
        self._external = external

    def write(self, trace: AgentTrace) -> None:
        self._memory.write(trace)
        if self._external is not None:
            self._external.write(trace)


def load_control_plane_ablation(manifest_path: Path) -> AblationEvalSet:
    raw = manifest_path.read_bytes()
    manifest = CombinedManifest.model_validate_json(raw)
    root = manifest_path.parents[2]
    sources = {item.role: item for item in manifest.sources}
    required_roles = {
        "phase61_scenarios", "phase61_expectations", "phase63_scenarios",
        "phase63_expectations", "counterfactuals",
    }
    if set(sources) != required_roles:
        raise ValueError("Combined manifest has unexpected source roles")
    resolved = {}
    for role, source in sources.items():
        path = (root / source.path).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Combined manifest source escapes repository root")
        if sha256(path.read_bytes()).hexdigest() != source.sha256:
            raise ValueError(f"Combined manifest hash mismatch for {role}")
        resolved[role] = path
    phase61 = load_e2e_eval_set(resolved["phase61_scenarios"], resolved["phase61_expectations"])
    phase63 = load_e2e_eval_set(resolved["phase63_scenarios"], resolved["phase63_expectations"])
    scenarios = [*phase61[0], *phase63[0]]
    expectations = [*phase61[1], *phase63[1]]
    ordered = [item.case_id for item in manifest.cases]
    if ordered != [item.case_id for item in scenarios] or ordered != [item.case_id for item in expectations]:
        raise ValueError("Combined manifest case order does not match datasets")
    rules = _load_rules(resolved["counterfactuals"])
    known = set(ordered)
    if any(item.case_id not in known for item in rules):
        raise ValueError("Counterfactual rule references an unknown case")
    rule_keys = [(x.case_id, x.component, x.trigger, x.call_index) for x in rules]
    if len(rule_keys) != len(set(rule_keys)):
        raise ValueError("Counterfactual rules must be unique")
    return AblationEvalSet(
        manifest_hash=sha256(raw).hexdigest(), manifest=manifest,
        scenarios=scenarios, expectations=expectations, counterfactuals=rules,
    )


def evaluate_control_plane_ablation(
    eval_set: AblationEvalSet, *, trace_sink: TraceSink | None = None,
) -> ControlPlaneAblationReport:
    mechanisms = {item.case_id: item.mechanism for item in eval_set.manifest.cases}
    cells = []
    for configuration in ABLATION_CONFIGURATIONS:
        for scenario, expectation in zip(eval_set.scenarios, eval_set.expectations):
            rules = [item for item in eval_set.counterfactuals if item.case_id == scenario.case_id]
            cells.append(_run_cell(
                scenario, expectation, mechanisms[scenario.case_id], configuration, rules, trace_sink,
            ))
    metrics = [_aggregate(configuration.name, cells) for configuration in ABLATION_CONFIGURATIONS]
    deltas = [_delta(before, after) for before, after in zip(metrics, metrics[1:])]
    attributions = _attribute(cells, eval_set.expectations)
    expected_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    staircase = [item.recovery_rate for item in metrics] == expected_rates
    phase61_ids = {item.case_id for item in eval_set.scenarios[:12]}
    expectations = {item.case_id: item for item in eval_set.expectations}
    full_compatible = all(
        _full_cell_compatible(cell, expectations[cell.case_id])
        for cell in cells
        if cell.configuration == "full_agent" and cell.case_id in phase61_ids
    )
    all_passed = (
        full_compatible and staircase
        and all(item.status == "credited" for item in attributions)
        and all(item.trace_health == "healthy" for item in cells)
    )
    return ControlPlaneAblationReport(
        manifest_hash=eval_set.manifest_hash, case_count=len(eval_set.scenarios),
        run_count=len(cells), metrics=metrics, deltas=deltas,
        attributions=attributions, cells=cells,
        phase61_full_compatible=full_compatible, recovery_staircase_valid=staircase,
        all_passed=all_passed,
    )


def _run_cell(scenario, expectation, mechanism, configuration, rules, external_sink):
    registry, call_context = _fixture_registry(scenario.user_faults)
    context_manager = ContextManager() if configuration.history_context_enabled else NoHistoryContextAdapter()
    planner = StructuredPlanner(
        _AblationReplayProvider(scenario.case_id, "planner", scenario.planner_calls, rules), registry,
        context_manager=context_manager, context_policy=scenario.context_policy,
    )
    writer = AnswerWriter(
        _AblationReplayProvider(scenario.case_id, "writer", scenario.answer_calls, rules), registry,
        context_manager=context_manager, context_policy=scenario.context_policy,
    )
    verifier = (
        StructuredVerifier(
            _AblationReplayProvider(scenario.case_id, "verifier", scenario.verifier_calls, rules),
            registry, context_manager=context_manager, context_policy=scenario.context_policy,
        )
        if configuration.verifier_rewrite_enabled else _PassThroughVerifier()
    )
    loop_policy = scenario.loop_policy.model_copy(update={
        "max_rewrite": scenario.loop_policy.max_rewrite if configuration.verifier_rewrite_enabled else 0,
        "max_replan": scenario.loop_policy.max_replan if configuration.replan_enabled else 0,
    })
    retry_policy = scenario.retry_policy if configuration.retry_enabled else scenario.retry_policy.model_copy(
        update={"max_retry": 0},
    )
    memory = InMemoryTraceSink()
    result = None
    exception_type = None
    try:
        traced = run_agent_loop_traced(
            UserQuery(query=scenario.query, history=scenario.history),
            planner, PlanValidator(registry), registry, writer, verifier,
            policy=loop_policy, retry_policy=retry_policy, context=call_context,
            max_concurrency=1, sleeper=lambda _: None, capture_mode="evaluation",
            sink=_TeeSink(memory, external_sink),
        )
        result = traced.result
        trace = traced.trace
    except BaseException as exc:
        exception_type = type(exc).__name__
        if not memory.traces:
            raise AssertionError("Exceptional ablation run did not produce an audit trace") from exc
        trace = memory.traces[-1]
    attempts = [item for item in trace.events if isinstance(item, ToolAttemptEvent)]
    invocations = [ToolInvocation(
        tool_name=item.tool_name, arguments=item.arguments.value,
        status=item.result_status, error_code=item.error_code,
    ) for item in attempts]
    actual_tools = _logical_tools(invocations)
    expected_tools = [_normalize_tool(registry, item) for item in expectation.expected_tools]
    actual_counter = Counter(_tool_key(item) for item in actual_tools)
    expected_counter = Counter(_tool_key(item) for item in expected_tools)
    unnecessary = sum((actual_counter - expected_counter).values())
    answer_correct = bool(result and _answer_correct(result, expectation, registry, actual_tools))
    results_ok = bool(result and all(item.result.status != "error" for item in result.task_results))
    task_success = bool(result and result.status == "completed" and answer_correct and results_ok)
    contexts = [item for item in trace.events if isinstance(item, ContextSelectedEvent)]
    breakdown = {component: 0 for component in ("planner", "writer", "verifier")}
    for item in contexts:
        breakdown[item.component] += item.metrics.selected_tokens
    loop_iterations = len([
        item for item in trace.events if isinstance(item, LoopIterationCompletedEvent)
    ])
    failure_type = _failure_type(result, trace, answer_correct, exception_type)
    terminal = [
        item for item in trace.events if isinstance(item, (RunFinishedEvent, RunFailedEvent))
    ]
    latency = terminal[-1].elapsed_ms if terminal else max((item.elapsed_ms for item in trace.events), default=0)
    trace_health = "degraded" if (
        trace.dropped_event_count or trace.recorder_error_codes
        or any(item.kind == "trace_degraded" for item in trace.events)
    ) else "healthy"
    return AblationCellResult(
        case_id=scenario.case_id, configuration=configuration.name, mechanism=mechanism,
        solvable=expectation.solvable, task_success=task_success, answer_correct=answer_correct,
        tool_selection_correct=actual_counter == expected_counter,
        unnecessary_tool_calls=unnecessary, logical_tool_count=len(actual_tools),
        tool_attempts=len(attempts), loop_iterations=loop_iterations,
        total_selected_history_tokens=sum(breakdown.values()),
        selected_history_tokens_by_component=breakdown, latency_ms=latency,
        run_failure_type=failure_type, trace_health=trace_health,
        exception_type=exception_type, result=result, trace=trace,
    )


def _failure_type(result, trace, answer_correct, exception_type) -> RunFailureType:
    if exception_type is not None or any(isinstance(item, RunFailedEvent) for item in trace.events):
        return "component_exception"
    blocked = [item for item in trace.events if isinstance(item, ToolAttemptBlockedEvent)]
    if any(item.reason == "deadline" for item in blocked):
        return "deadline_exhausted"
    if result is not None and result.stop_reason == "total_tool_budget":
        return "tool_budget_exhausted"
    if result is not None and result.stop_reason in {
        "max_iterations", "max_rewrite", "max_replan", "no_progress",
    }:
        return result.stop_reason
    attempts = [item for item in trace.events if isinstance(item, ToolAttemptEvent)]
    latest = {}
    for item in attempts:
        latest[item.task_ref] = item
    if any(item.result_status == "error" and item.retryable for item in latest.values()):
        return "retryable_tool_error"
    if any(item.result_status == "error" for item in latest.values()):
        return "non_retryable_tool_error"
    if result is not None and result.status in {"clarify", "no_tool"}:
        return result.status
    if result is not None and any(item.result.status == "error" for item in result.task_results):
        return "task_result_error"
    if not answer_correct:
        return "incorrect_answer"
    if result is not None and result.status == "completed":
        return "success"
    return "unclassified"


def _aggregate(configuration, cells):
    selected = [item for item in cells if item.configuration == configuration]
    solvable = [item for item in selected if item.solvable]
    recovery = [item for item in selected if item.mechanism in {"retry", "rewrite", "replan", "context"}]
    attempts = sum(item.tool_attempts for item in selected)
    logical = sum(item.logical_tool_count for item in selected)
    latencies = sorted(item.latency_ms for item in selected)
    count = len(selected)
    failure_counts = Counter(item.run_failure_type for item in selected)
    health_counts = Counter(item.trace_health for item in selected)
    return AblationMetrics(
        configuration=configuration,
        task_success_rate=_rate(sum(item.task_success for item in solvable), len(solvable)),
        final_answer_correctness=_rate(sum(item.answer_correct for item in solvable), len(solvable)),
        recovery_rate=_rate(sum(item.task_success for item in recovery), len(recovery)),
        total_tool_attempts=attempts, average_tool_attempts=attempts / count,
        total_unnecessary_tool_calls=sum(item.unnecessary_tool_calls for item in selected),
        unnecessary_tool_call_rate=_rate(
            sum(item.unnecessary_tool_calls for item in selected), logical,
        ),
        average_loop_iterations=sum(item.loop_iterations for item in selected) / count,
        average_total_selected_history_tokens=(
            sum(item.total_selected_history_tokens for item in selected) / count
        ),
        average_selected_history_tokens_by_component={
            component: sum(item.selected_history_tokens_by_component[component] for item in selected) / count
            for component in ("planner", "writer", "verifier")
        },
        latency_mean_ms=sum(latencies) / count,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        run_failure_types=dict(sorted(failure_counts.items())),
        trace_health=dict(sorted(health_counts.items())),
    )


def _delta(before: AblationMetrics, after: AblationMetrics) -> AblationDelta:
    return AblationDelta(
        before_config=before.configuration, after_config=after.configuration,
        task_success_rate_delta=after.task_success_rate - before.task_success_rate,
        final_answer_correctness_delta=(
            after.final_answer_correctness - before.final_answer_correctness
        ),
        recovery_rate_delta=after.recovery_rate - before.recovery_rate,
        average_tool_attempts_delta=after.average_tool_attempts - before.average_tool_attempts,
        unnecessary_tool_call_rate_delta=(
            after.unnecessary_tool_call_rate - before.unnecessary_tool_call_rate
        ),
        average_loop_iterations_delta=(
            after.average_loop_iterations - before.average_loop_iterations
        ),
        average_total_selected_history_tokens_delta=(
            after.average_total_selected_history_tokens
            - before.average_total_selected_history_tokens
        ),
        latency_mean_ms_delta=after.latency_mean_ms - before.latency_mean_ms,
    )


def _attribute(cells, expectations):
    by_key = {(item.configuration, item.case_id): item for item in cells}
    expected = {item.case_id: item for item in expectations}
    transitions = [
        ("retry_only_recovery", "retry", "simple_baseline", "retry"),
        ("rewrite_only_recovery", "rewrite", "retry", "verifier_rewrite"),
        ("replan_only_recovery", "replan", "verifier_rewrite", "replan"),
        ("context_only_recovery", "context", "replan", "full_agent"),
    ]
    values = []
    for case_id, mechanism, before_name, after_name in transitions:
        before, after = by_key[(before_name, case_id)], by_key[(after_name, case_id)]
        evidence = _attribution_evidence(mechanism, before, after, expected[case_id])
        if before.task_success and after.task_success:
            status = "already_solved"
        elif before.task_success and not after.task_success:
            status = "regression"
        elif not after.task_success:
            status = "not_recovered"
        else:
            status = "credited" if evidence else "not_recovered"
        values.append(MechanismAttribution(
            case_id=case_id, mechanism=mechanism, before_config=before_name,
            after_config=after_name, before_trace_id=str(before.trace.trace_id),
            after_trace_id=str(after.trace.trace_id),
            before_failure_type=before.run_failure_type,
            after_failure_type=after.run_failure_type,
            status=status, evidence=evidence,
        ))
    return values


def _attribution_evidence(mechanism, before, after, expectation):
    events = after.trace.events
    if mechanism == "retry":
        for failed in events:
            if not isinstance(failed, ToolAttemptEvent) or not failed.retryable or failed.result_status != "error":
                continue
            scheduled = next((item for item in events if isinstance(item, RetryScheduledEvent)
                              and item.task_ref == failed.task_ref and item.sequence > failed.sequence), None)
            success = next((item for item in events if isinstance(item, ToolAttemptEvent)
                            and item.task_ref == failed.task_ref and item.result_status != "error"
                            and item.sequence > (scheduled.sequence if scheduled else failed.sequence)), None)
            if scheduled and success:
                return [_pointer(failed), _pointer(scheduled), _pointer(success)]
    if mechanism == "rewrite":
        rewrite = next((item for item in events if isinstance(item, VerifierCompletedEvent)
                        and item.decision == "REWRITE"), None)
        writer = next((item for item in events if isinstance(item, WriterCompletedEvent)
                       and item.mode == "rewrite" and rewrite and item.sequence > rewrite.sequence), None)
        passed = next((item for item in events if isinstance(item, VerifierCompletedEvent)
                       and item.decision == "PASS" and writer and item.sequence > writer.sequence), None)
        tool_between = any(isinstance(item, ToolAttemptEvent) and rewrite and passed
                           and rewrite.sequence < item.sequence < passed.sequence for item in events)
        if rewrite and writer and passed and not tool_between:
            return [_pointer(rewrite), _pointer(writer), _pointer(passed)]
    if mechanism == "replan":
        replan = next((item for item in events if isinstance(item, VerifierCompletedEvent)
                       and item.decision == "REPLAN"), None)
        plan = next((item for item in events if isinstance(item, PlanProposedEvent)
                     and replan and item.sequence > replan.sequence and (item.plan_revision or 0) > 0), None)
        passed = next((item for item in events if isinstance(item, VerifierCompletedEvent)
                       and item.decision == "PASS" and plan and item.sequence > plan.sequence), None)
        if replan and plan and passed:
            return [_pointer(replan), _pointer(plan), _pointer(passed)]
    if mechanism == "context":
        required = set(expectation.required_retrieved_message_indexes)
        before_context = [item for item in before.trace.events
                          if isinstance(item, ContextSelectedEvent) and item.component == "planner"]
        after_context = [item for item in events
                         if isinstance(item, ContextSelectedEvent) and item.component == "planner"]
        target = next((item for item in after_context if item.retrieval_active
                       and required.issubset(item.retrieved_message_indexes)), None)
        absent_before = all(required.isdisjoint(item.selected_message_indexes)
                            and required.isdisjoint(item.retrieved_message_indexes)
                            for item in before_context)
        plan = next((item for item in events if isinstance(item, PlanProposedEvent)
                     and target and item.sequence > target.sequence and item.decision == "execute"), None)
        if target and plan and absent_before:
            return [_pointer(target), _pointer(plan)]
    return []


def _pointer(event):
    return TraceEventPointer(
        kind=event.kind, sequence=event.sequence, task_ref=getattr(event, "task_ref", None),
    )


def _full_cell_compatible(cell, expectation):
    if cell.result is None:
        return False
    retrieved = {
        index for item in cell.trace.events if isinstance(item, ContextSelectedEvent)
        and item.retrieval_active for index in item.retrieved_message_indexes
    }
    return all((
        cell.result.status == expectation.status,
        cell.result.stop_reason == expectation.stop_reason,
        cell.tool_attempts == expectation.expected_tool_attempts,
        cell.loop_iterations == expectation.expected_loop_iterations,
        cell.result.rewrite_count == expectation.expected_rewrite_count,
        cell.result.replan_count == expectation.expected_replan_count,
        (cell.answer_correct if expectation.solvable else True),
        cell.tool_selection_correct == expectation.expected_tool_selection_correct,
        cell.unnecessary_tool_calls == expectation.expected_unnecessary_calls,
        set(expectation.required_retrieved_message_indexes).issubset(retrieved),
    ))


def _load_rules(path):
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(CounterfactualRule.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid counterfactual rule at {path}:{number}") from exc
    return values


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 1.0


def _percentile(values, quantile):
    if not values:
        return 0.0
    return values[max(0, ceil(len(values) * quantile) - 1)]
