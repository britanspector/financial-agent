"""Versioned, strongly typed contracts for run-scoped Agent traces."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, TypeAdapter

from financial_agent.context.models import ContextMetrics
from financial_agent.schemas import Schema


TraceCaptureMode = Literal["safe", "evaluation"]


class ValueProjection(Schema):
    """A capture-mode-aware projection of a public value."""

    capture_mode: TraceCaptureMode
    item_count: int = Field(ge=0)
    value_types: list[str] = Field(default_factory=list)
    digest: str = Field(min_length=1)
    value: Any | None = Field(default=None, exclude_if=lambda value: value is None)


class BindingProjection(Schema):
    target_ref: str
    source_task_ref: str
    source_path_ref: str
    target_parameter: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_path: list[str | int] | None = Field(default=None, exclude_if=lambda value: value is None)


class TaskProjection(Schema):
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str
    arguments: ValueProjection
    dependency_refs: list[str] = Field(default_factory=list)
    dependency_ids: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    bindings: list[BindingProjection] = Field(default_factory=list)


class EvidenceProjection(Schema):
    task_ref: str
    path_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_path: list[str | int] | None = Field(default=None, exclude_if=lambda value: value is None)


class BudgetProjection(Schema):
    limit: int = Field(ge=1)
    used: int = Field(ge=0)
    remaining: int = Field(ge=0)
    exhausted: bool
    deadline_exceeded: bool = False


class TraceEventBase(Schema):
    sequence: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    iteration: int | None = Field(default=None, ge=0)
    plan_revision: int | None = Field(default=None, ge=0)
    operation: str | None = None


class RunStartedEvent(TraceEventBase):
    kind: Literal["run_started"] = "run_started"
    request_ref: str
    query_ref: str
    query: str | None = Field(default=None, exclude_if=lambda value: value is None)
    history: list[dict[str, str]] | None = Field(default=None, exclude_if=lambda value: value is None)
    policy: ValueProjection


class ContextSelectedEvent(TraceEventBase):
    kind: Literal["context_selected"] = "context_selected"
    component: Literal["planner", "writer", "verifier"]
    metrics: ContextMetrics
    selected_message_indexes: list[int] = Field(default_factory=list)
    retrieved_message_indexes: list[int] = Field(default_factory=list)
    summary_present: bool
    retrieval_active: bool
    selected_content: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    retrieved_content: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)


class PlanProposedEvent(TraceEventBase):
    kind: Literal["plan_proposed"] = "plan_proposed"
    decision: Literal["execute", "clarify", "no_tool"]
    tasks: list[TaskProjection] = Field(default_factory=list)
    force_rerun_refs: list[str] = Field(default_factory=list)
    force_rerun_task_ids: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)


class PlanValidatedEvent(TraceEventBase):
    kind: Literal["plan_validated"] = "plan_validated"
    valid: bool
    decision: Literal["execute", "clarify", "no_tool"] | None = None
    task_refs: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)


class ToolReuseEvent(TraceEventBase):
    kind: Literal["tool_reuse"] = "tool_reuse"
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str


class ToolAttemptEvent(TraceEventBase):
    kind: Literal["tool_attempt"] = "tool_attempt"
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str
    attempt_index: int = Field(ge=1)
    arguments: ValueProjection
    result_status: Literal["success", "empty", "error"]
    result: ValueProjection | None = None
    error_code: str | None = None
    retryable: bool = False
    exception_type: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    budget_before: BudgetProjection
    budget_after: BudgetProjection


class ToolAttemptBlockedEvent(TraceEventBase):
    kind: Literal["tool_attempt_blocked"] = "tool_attempt_blocked"
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str
    attempt_index: int = Field(ge=1)
    reason: Literal["attempt_budget", "deadline"]
    budget: BudgetProjection


class RetryScheduledEvent(TraceEventBase):
    kind: Literal["retry_scheduled"] = "retry_scheduled"
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str
    next_attempt_index: int = Field(ge=2)
    backoff_seconds: float = Field(ge=0, allow_inf_nan=False)
    budget: BudgetProjection


class RetrySkippedEvent(TraceEventBase):
    kind: Literal["retry_skipped"] = "retry_skipped"
    task_ref: str
    task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str
    next_attempt_index: int = Field(ge=2)
    reason: Literal["not_retryable", "max_retry", "attempt_budget", "deadline"]
    budget: BudgetProjection


class WriterCompletedEvent(TraceEventBase):
    kind: Literal["writer_completed"] = "writer_completed"
    mode: Literal["write", "rewrite", "initial_draft"]
    answer_ref: str
    answer: str | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence: list[EvidenceProjection] = Field(default_factory=list)


class VerifierCompletedEvent(TraceEventBase):
    kind: Literal["verifier_completed"] = "verifier_completed"
    decision: Literal["PASS", "REWRITE", "REPLAN"]
    failed_task_refs: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    reason_ref: str
    reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    missing_evidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)


class LoopIterationCompletedEvent(TraceEventBase):
    kind: Literal["loop_iteration_completed"] = "loop_iteration_completed"
    action: Literal["initial", "rewrite", "replan"]
    decision: Literal["PASS", "REWRITE", "REPLAN"]
    rewrite_count: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    budget: BudgetProjection


class ComponentFailedEvent(TraceEventBase):
    kind: Literal["component_failed"] = "component_failed"
    component: str
    error_code: str
    exception_type: str


class RunFinishedEvent(TraceEventBase):
    kind: Literal["run_finished"] = "run_finished"
    status: Literal["completed", "limit_exhausted", "clarify", "no_tool"]
    stop_reason: str
    rewrite_count: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    iteration_count: int = Field(ge=0)
    budget: BudgetProjection
    answer_ref: str | None = None
    final_answer: str | None = Field(default=None, exclude_if=lambda value: value is None)


class RunFailedEvent(TraceEventBase):
    kind: Literal["run_failed"] = "run_failed"
    error_code: str
    exception_type: str


class TraceDegradedEvent(TraceEventBase):
    kind: Literal["trace_degraded"] = "trace_degraded"
    error_code: str
    dropped_event_count: int = Field(ge=1)


TraceEvent = Annotated[
    RunStartedEvent | ContextSelectedEvent | PlanProposedEvent | PlanValidatedEvent
    | ToolReuseEvent | ToolAttemptEvent | ToolAttemptBlockedEvent | RetryScheduledEvent
    | RetrySkippedEvent | WriterCompletedEvent | VerifierCompletedEvent
    | LoopIterationCompletedEvent | ComponentFailedEvent | RunFinishedEvent
    | RunFailedEvent | TraceDegradedEvent,
    Field(discriminator="kind"),
]
TRACE_EVENT_ADAPTER = TypeAdapter(TraceEvent)


class AgentTrace(Schema):
    schema_version: Literal["1.0"] = "1.0"
    trace_id: UUID
    request_id: UUID
    capture_mode: TraceCaptureMode
    events: list[TraceEvent] = Field(default_factory=list)
    completion_status: Literal["completed", "failed", "degraded"]
    dropped_event_count: int = Field(default=0, ge=0)
    recorder_error_codes: list[str] = Field(default_factory=list)


class TracePersistence(Schema):
    status: Literal["not_requested", "saved", "failed"]
    error_code: str | None = None
