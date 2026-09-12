"""Structured planner input/output and deterministic validation contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from financial_agent.agent.models import Task
from financial_agent.agent.models import ResultBinding
from financial_agent.schemas import Schema


class PlannedTask(Schema):
    """LLM-facing task shape; graph validity is deliberately checked later."""

    task_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    bindings: list[ResultBinding] = Field(default_factory=list)


class StructuredPlan(Schema):
    decision: Literal["execute", "clarify", "no_tool"]
    tasks: list[PlannedTask]


class PlanValidationIssue(Schema):
    code: Literal[
        "EMPTY_PLAN",
        "NON_EXECUTION_HAS_TASKS",
        "TOO_MANY_TASKS",
        "DUPLICATE_TASK_ID",
        "UNKNOWN_TOOL",
        "INVALID_ARGUMENTS",
        "DYNAMIC_RESULT_REFERENCE",
        "INVALID_BINDING_TASK",
        "SELF_BINDING",
        "DUPLICATE_BINDING_TARGET",
        "INVALID_BINDING_FIELD",
        "BINDING_TYPE_MISMATCH",
        "MISSING_DEPENDENCY",
        "SELF_DEPENDENCY",
        "DUPLICATE_DEPENDENCY",
        "DEPENDENCY_CYCLE",
    ]
    message: str
    task_id: str | None = None


class PlanValidationResult(Schema):
    valid: bool
    decision: Literal["execute", "clarify", "no_tool"] | None = None
    tasks: list[Task] = Field(default_factory=list)
    issues: list[PlanValidationIssue] = Field(default_factory=list)
