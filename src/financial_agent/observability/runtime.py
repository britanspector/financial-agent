"""Compatibility-preserving traced entry point around the production loop."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep
from uuid import UUID

from financial_agent.agent.loop import AgentLoopResult, LoopPolicy, run_agent_loop
from financial_agent.agent.retry import RetryPolicy
from financial_agent.observability.models import AgentTrace, TraceCaptureMode, TracePersistence
from financial_agent.observability.recorder import AgentTraceRecorder
from financial_agent.observability.sinks import TraceSink
from financial_agent.schemas import Schema, UserQuery
from financial_agent.user_data.auth import CallContext
from financial_agent.verifier.models import DraftAnswer


class TracedAgentLoopResult(Schema):
    result: AgentLoopResult
    trace: AgentTrace
    persistence: TracePersistence


def run_agent_loop_traced(
    request: UserQuery, planner, validator, registry, writer, verifier, *,
    policy: LoopPolicy | None = None,
    retry_policy: RetryPolicy | None = None,
    initial_draft: DraftAnswer | None = None,
    context: CallContext | None = None,
    max_concurrency: int | None = None,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
    capture_mode: TraceCaptureMode = "safe",
    sink: TraceSink | None = None,
    trace_id: UUID | None = None,
    trace_clock: Callable[[], float] = monotonic,
    recorder: AgentTraceRecorder | None = None,
) -> TracedAgentLoopResult:
    limits = policy or LoopPolicy()
    retry = retry_policy or RetryPolicy()
    audit = recorder or AgentTraceRecorder(
        request.request_id, capture_mode=capture_mode, trace_id=trace_id, clock=trace_clock,
    )
    try:
        with audit.activate(), audit.scope(operation="run", iteration=0, plan_revision=0):
            audit.record_run_started(request, limits, retry)
            result = run_agent_loop(
                request, planner, validator, registry, writer, verifier,
                policy=limits, retry_policy=retry, initial_draft=initial_draft,
                context=context, max_concurrency=max_concurrency, clock=clock, sleeper=sleeper,
            )
            audit.record_finished(result, limits.total_tool_budget)
        trace = _finalize_safely(audit)
    except BaseException as exc:
        try:
            audit.record_component_failed("agent_loop", exc)
            audit.record_failed(exc)
            trace = _finalize_safely(audit, failed=True)
            _persist(trace, sink)
        except BaseException:
            pass
        raise
    persistence = _persist(trace, sink)
    return TracedAgentLoopResult(result=result, trace=trace, persistence=persistence)


def _persist(trace: AgentTrace, sink: TraceSink | None) -> TracePersistence:
    if sink is None:
        return TracePersistence(status="not_requested")
    try:
        sink.write(trace)
    except BaseException:
        return TracePersistence(status="failed", error_code="TRACE_SINK_FAILED")
    return TracePersistence(status="saved")


def _finalize_safely(audit: AgentTraceRecorder, *, failed: bool = False) -> AgentTrace:
    try:
        return audit.finalize(failed=failed)
    except BaseException:
        return AgentTrace(
            trace_id=audit.trace_id, request_id=audit.request_id,
            capture_mode=audit.capture_mode, events=[],
            completion_status="failed" if failed else "degraded",
            dropped_event_count=1, recorder_error_codes=["TRACE_FINALIZATION_FAILED"],
        )
