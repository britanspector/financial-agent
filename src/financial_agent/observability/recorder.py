"""Failure-isolated recorder and synchronous run scope."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Callable, Iterator
from uuid import UUID, uuid4

from financial_agent.agent.models import Task
from financial_agent.observability.models import (
    AgentTrace, ComponentFailedEvent, ContextSelectedEvent, LoopIterationCompletedEvent,
    ModelCallCompletedEvent, ModelCallFailedEvent, ModelCallStartedEvent,
    PlanProposedEvent, PlanValidatedEvent, RetryScheduledEvent, RetrySkippedEvent,
    RunFailedEvent, RunFinishedEvent, RunStartedEvent, ToolAttemptBlockedEvent,
    ToolAttemptEvent, ToolReuseEvent, TraceCaptureMode, TraceDegradedEvent, TraceEvent,
    VerifierCompletedEvent, WriterCompletedEvent,
)
from financial_agent.observability.projection import TraceProjector, TraceSanitizer


@dataclass(frozen=True)
class TraceScope:
    operation: str | None = None
    iteration: int | None = None
    plan_revision: int | None = None


_ACTIVE_RECORDER: ContextVar[AgentTraceRecorder | None] = ContextVar("agent_trace_recorder", default=None)
_ACTIVE_SCOPE: ContextVar[TraceScope] = ContextVar("agent_trace_scope", default=TraceScope())


def active_recorder() -> AgentTraceRecorder | None:
    return _ACTIVE_RECORDER.get()


def active_scope() -> TraceScope:
    return _ACTIVE_SCOPE.get()


class AgentTraceRecorder:
    def __init__(
        self,
        request_id: UUID,
        *,
        capture_mode: TraceCaptureMode = "safe",
        trace_id: UUID | None = None,
        clock: Callable[[], float] = monotonic,
        projector: TraceProjector | None = None,
        sanitizer: TraceSanitizer | None = None,
    ) -> None:
        self.request_id = request_id
        self.trace_id = trace_id or uuid4()
        self.capture_mode = capture_mode
        self.projector = projector or TraceProjector(capture_mode, secrets.token_bytes(32))
        self.sanitizer = sanitizer or TraceSanitizer()
        self._clock = clock
        self._started = clock()
        self._events: list[TraceEvent] = []
        self._sequence = 0
        self._dropped = 0
        self._error_codes: list[str] = []
        self._model_call_counts: dict[str, int] = {}
        self._lock = Lock()

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _ACTIVE_RECORDER.set(self)
        try:
            yield
        finally:
            _ACTIVE_RECORDER.reset(token)

    @contextmanager
    def scope(
        self, *, operation: str | None = None, iteration: int | None = None,
        plan_revision: int | None = None,
    ) -> Iterator[None]:
        previous = active_scope()
        value = TraceScope(
            operation=operation if operation is not None else previous.operation,
            iteration=iteration if iteration is not None else previous.iteration,
            plan_revision=plan_revision if plan_revision is not None else previous.plan_revision,
        )
        token = _ACTIVE_SCOPE.set(value)
        try:
            yield
        finally:
            _ACTIVE_SCOPE.reset(token)

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def _record(self, event_type, **fields: Any) -> None:
        try:
            scope = active_scope()
            with self._lock:
                sequence = self._sequence + 1
                kind = event_type.model_fields["kind"].default
                fields = self.projector.event_fields(kind, fields)
                event = event_type(
                    sequence=sequence,
                    elapsed_ms=max(0.0, (self._clock() - self._started) * 1000),
                    iteration=fields.pop("iteration", scope.iteration),
                    plan_revision=fields.pop("plan_revision", scope.plan_revision),
                    operation=fields.pop("operation", scope.operation),
                    **fields,
                )
                self.sanitizer.validate(event, self.capture_mode)
                self._sequence = sequence
                self._events.append(event)
        except BaseException:
            self._degrade("TRACE_EVENT_DROPPED")

    def _degrade(self, code: str) -> None:
        with self._lock:
            self._dropped += 1
            if code not in self._error_codes:
                self._error_codes.append(code)

    def record_run_started(self, request, policy, retry_policy) -> None:
        try:
            query_ref, query = self.projector.text(request.query)
            history = (
                [item.model_dump(mode="json") for item in request.history]
                if self.capture_mode == "evaluation" else None
            )
            self._record(RunStartedEvent, request_ref=self.projector.reference(request.request_id),
                         query_ref=query_ref, query=query, history=history,
                         policy=self.projector.value({"loop": policy, "retry": retry_policy}))
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_context(self, selection, selected_indexes: list[int]) -> None:
        try:
            retrieved_indexes = sorted({
                index for turn in selection.retrieved_history for index in turn.message_indexes
            })
            self._record(
                ContextSelectedEvent,
                component=selection.metrics.component,
                metrics=selection.metrics,
                selected_message_indexes=selected_indexes,
                retrieved_message_indexes=retrieved_indexes,
                summary_present=selection.summary is not None,
                retrieval_active=selection.metrics.retrieval_active,
                selected_content=(
                    [item.content for item in selection.request.history]
                    if self.capture_mode == "evaluation" else None
                ),
                retrieved_content=(
                    [message.content for turn in selection.retrieved_history for message in turn.messages]
                    if self.capture_mode == "evaluation" else None
                ),
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_plan(self, output, *, force_rerun_ids: list[str] | None = None) -> None:
        try:
            tasks = [self.projector.task(task) for task in output.tasks]
            forced = list(force_rerun_ids or [])
            self._record(
                PlanProposedEvent, decision=getattr(output, "decision", "execute"), tasks=tasks,
                force_rerun_refs=[self.projector.reference(item) for item in forced],
                force_rerun_task_ids=forced if self.capture_mode == "evaluation" else None,
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_model_call_started(self, component: str, model: str, messages, response_schema) -> int:
        with self._lock:
            call_index = self._model_call_counts.get(component, 0) + 1
            self._model_call_counts[component] = call_index
        self._record(
            ModelCallStartedEvent,
            component=component,
            call_index=call_index,
            model=model,
            messages=self.projector.value(messages),
            response_schema=self.projector.value(response_schema),
        )
        return call_index

    def record_model_call_completed(self, component: str, call_index: int, model: str, response) -> None:
        self._record(
            ModelCallCompletedEvent,
            component=component,
            call_index=call_index,
            model=model,
            response=self.projector.value(response),
        )

    def record_model_call_failed(
        self, component: str, call_index: int, model: str, exc: BaseException,
    ) -> None:
        self._record(
            ModelCallFailedEvent,
            component=component,
            call_index=call_index,
            model=model,
            exception_type=type(exc).__name__,
        )

    def record_validation(self, validation) -> None:
        try:
            self._record(
                PlanValidatedEvent, valid=validation.valid, decision=validation.decision,
                task_refs=[self.projector.reference(item.task_id) for item in validation.tasks],
                issue_codes=[item.code for item in validation.issues],
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_reuse(self, task: Task) -> None:
        try:
            self._record(
                ToolReuseEvent, task_ref=self.projector.reference(task.task_id),
                task_id=task.task_id if self.capture_mode == "evaluation" else None,
                tool_name=task.tool_name,
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_tool_attempt(
        self, task: Task, attempt_index: int, result, latency_ms: float,
        budget_before, budget_after,
    ) -> None:
        try:
            error = result.error
            self._record(
                ToolAttemptEvent, task_ref=self.projector.reference(task.task_id),
                task_id=task.task_id if self.capture_mode == "evaluation" else None,
                tool_name=task.tool_name, attempt_index=attempt_index,
                arguments=self.projector.value(task.arguments), result_status=result.status,
                result=self.projector.result(result.data),
                error_code=error.code if error else None,
                retryable=error.retryable if error else False, latency_ms=latency_ms,
                budget_before=budget_before, budget_after=budget_after,
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_tool_blocked(self, task: Task, attempt_index: int, reason: str, budget) -> None:
        try:
            self._record(
                ToolAttemptBlockedEvent, task_ref=self.projector.reference(task.task_id),
                task_id=task.task_id if self.capture_mode == "evaluation" else None,
                tool_name=task.tool_name, attempt_index=attempt_index, reason=reason, budget=budget,
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_retry(self, task: Task, next_index: int, delay: float, budget, *, reason=None) -> None:
        try:
            common = dict(
                task_ref=self.projector.reference(task.task_id),
                task_id=task.task_id if self.capture_mode == "evaluation" else None,
                tool_name=task.tool_name, next_attempt_index=next_index, budget=budget,
            )
            if reason is None:
                self._record(RetryScheduledEvent, **common, backoff_seconds=delay)
            else:
                self._record(RetrySkippedEvent, **common, reason=reason)
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_writer(self, draft, mode: str) -> None:
        try:
            answer_ref, answer = self.projector.text(draft.answer)
            self._record(WriterCompletedEvent, mode=mode, answer_ref=answer_ref, answer=answer,
                         evidence=[self.projector.evidence(item) for item in draft.evidence])
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_verifier(self, result) -> None:
        try:
            reason_ref, reason = self.projector.text(result.reason)
            self._record(
                VerifierCompletedEvent, decision=result.decision,
                failed_task_refs=[self.projector.reference(item) for item in result.failed_task_ids],
                failed_task_ids=(list(result.failed_task_ids) if self.capture_mode == "evaluation" else None),
                reason_ref=reason_ref, reason=reason,
                missing_evidence_refs=[self.projector.reference(item) for item in result.missing_evidence],
                missing_evidence=(list(result.missing_evidence) if self.capture_mode == "evaluation" else None),
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_iteration(self, action, decision, rewrites, replans, budget) -> None:
        try:
            self._record(LoopIterationCompletedEvent, action=action, decision=decision,
                         rewrite_count=rewrites, replan_count=replans, budget=self.projector.budget(budget))
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def record_component_failed(self, component: str, exc: BaseException) -> None:
        self._record(ComponentFailedEvent, component=component, error_code="COMPONENT_FAILED",
                     exception_type=type(exc).__name__)

    def record_finished(self, result, budget_limit: int) -> None:
        try:
            answer_ref = self.projector.reference(result.answer) if result.answer is not None else None
            final_answer = result.answer if self.capture_mode == "evaluation" else None
            budget = self._latest_budget() or _final_budget(budget_limit, result.total_tool_attempts)
            self._record(
                RunFinishedEvent, status=result.status, stop_reason=result.stop_reason,
                rewrite_count=result.rewrite_count, replan_count=result.replan_count,
                iteration_count=result.iteration_count, budget=budget,
                answer_ref=answer_ref, final_answer=final_answer,
                iteration=result.iteration_count, plan_revision=result.replan_count, operation="run",
            )
        except BaseException:
            self._degrade("TRACE_PROJECTION_FAILED")

    def _latest_budget(self):
        with self._lock:
            found = []
            for event in reversed(self._events):
                budget = getattr(event, "budget_after", None) or getattr(event, "budget", None)
                if budget is not None:
                    found.append(budget)
            if not found:
                return None
            latest = found[0]
            return latest.model_copy(update={
                "deadline_exceeded": any(item.deadline_exceeded for item in found),
            })

    def record_failed(self, exc: BaseException) -> None:
        self._record(RunFailedEvent, error_code="AGENT_RUN_FAILED", exception_type=type(exc).__name__)

    def finalize(self, *, failed: bool = False) -> AgentTrace:
        with self._lock:
            if self._dropped:
                self._sequence += 1
                fields = self.projector.event_fields("trace_degraded", {
                    "error_code": self._error_codes[-1], "dropped_event_count": self._dropped,
                })
                degraded = TraceDegradedEvent(
                    sequence=self._sequence,
                    elapsed_ms=max(0.0, (self._clock() - self._started) * 1000),
                    **fields,
                )
                try:
                    self.sanitizer.validate(degraded, self.capture_mode)
                    self._events.append(degraded)
                except BaseException:
                    pass
            trace = AgentTrace(
                trace_id=self.trace_id, request_id=self.request_id, capture_mode=self.capture_mode,
                events=list(self._events),
                completion_status="failed" if failed else ("degraded" if self._dropped else "completed"),
                dropped_event_count=self._dropped, recorder_error_codes=list(self._error_codes),
            )
        self.sanitizer.validate(trace, self.capture_mode)
        return trace


def _final_budget(limit: int, used: int):
    from financial_agent.observability.models import BudgetProjection
    return BudgetProjection(limit=limit, used=used, remaining=max(0, limit - used), exhausted=used >= limit)
