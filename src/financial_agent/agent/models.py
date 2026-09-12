"""Structured task, result-pool, and LangGraph state contracts."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.contracts import ToolResult


class Task(Schema):
    task_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    bindings: list["ResultBinding"] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_dependencies(self):
        if self.task_id in self.dependencies:
            raise ValueError("Task cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("Task dependencies must be unique")
        return self


class ResultBinding(Schema):
    """Copy one explicitly-addressed upstream result value into a Tool argument.

    ``source_path`` is deliberately a list of field names/list indexes instead
    of an expression language.  It is evaluated below ``ToolResult.data``.
    """

    target_parameter: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_path: list[str | int] = Field(min_length=1)


class TaskExecutionResult(Schema):
    task_id: str
    tool_name: str
    result: ToolResult[Any]
    retry_count: int = Field(default=0, ge=0)
    max_retry: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def retry_count_within_limit(self):
        if self.retry_count > self.max_retry:
            raise ValueError("retry_count cannot exceed max_retry")
        return self


class AgentError(Schema):
    task_id: str
    code: str
    message: str
    http_status: int = Field(ge=400, le=599)
    retryable: bool
    reason: Literal["dependency_failed", "missing_dependency", "cycle_or_deadlock"] | None = None


class FinalResult(Schema):
    status: Literal["success", "partial", "failed"]
    query: str
    task_results: list[TaskExecutionResult]
    errors: list[AgentError]
    iteration_count: int = Field(ge=0)
    attempt_count: int = Field(default=0, ge=0)
    execution_duration_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    attempt_budget_exhausted: bool = False
    deadline_exceeded: bool = False


class AgentState(Schema):
    request_id: UUID = Field(default_factory=uuid4)
    query: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    tool_results: Annotated[list[TaskExecutionResult], operator.add] = Field(default_factory=list)
    errors: Annotated[list[AgentError], operator.add] = Field(default_factory=list)
    final_output: FinalResult | None = None
    iteration_count: int = Field(default=0, ge=0)
    status: Literal["pending", "completed", "failed"] = "pending"
    scheduled_tasks: list[Task] = Field(default_factory=list, exclude=True)
    dispatch_action: Literal["execute", "collect", "finalize"] = Field(default="finalize", exclude=True)
    collected_result_count: int = Field(default=0, ge=0, exclude=True)

    @model_validator(mode="after")
    def unique_task_ids(self):
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique")
        return self

    @classmethod
    def from_query(cls, request: UserQuery, tasks: list[Task]) -> "AgentState":
        return cls(
            request_id=request.request_id,
            query=request.query,
            history=request.history,
            tasks=tasks,
        )
