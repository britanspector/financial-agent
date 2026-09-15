"""Shared validation for evidence paths rooted at ToolResult.data."""

from collections import Counter
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.agent.result_path import ResultPathError, resolve_result_path, result_path_type
from financial_agent.verifier.models import EvidenceReference


class OutputCatalog(Protocol):
    def output_model(self, name: str) -> type[BaseModel] | None: ...


class InvalidEvidenceInputError(ValueError):
    pass


def validate_execution(
    plan: Sequence[Task],
    results: Sequence[TaskExecutionResult],
) -> dict[str, TaskExecutionResult]:
    tasks = list(plan)
    items = list(results)
    if not tasks:
        raise InvalidEvidenceInputError("A non-empty execute plan is required")
    task_counts = Counter(task.task_id for task in tasks)
    result_counts = Counter(item.task_id for item in items)
    if any(count != 1 for count in task_counts.values()):
        raise InvalidEvidenceInputError("Plan task IDs must be unique")
    if any(count != 1 for count in result_counts.values()):
        raise InvalidEvidenceInputError("Tool result task IDs must be unique")
    if set(task_counts) != set(result_counts):
        raise InvalidEvidenceInputError("Plan and Tool result task IDs must match exactly")
    result_by_id = {item.task_id: item for item in items}
    for task in tasks:
        if result_by_id[task.task_id].tool_name != task.tool_name:
            raise InvalidEvidenceInputError("Plan and Tool result names must match")
    return result_by_id


def resolve_evidence(
    references: Sequence[EvidenceReference],
    plan: Sequence[Task],
    results: Sequence[TaskExecutionResult],
    catalog: OutputCatalog,
) -> list[dict[str, Any]]:
    tasks = list(plan)
    result_by_id = validate_execution(tasks, results)
    task_by_id = {task.task_id: task for task in tasks}
    resolved = []
    for reference in references:
        task = task_by_id.get(reference.task_id)
        if task is None:
            raise InvalidEvidenceInputError(f"Evidence task does not exist: {reference.task_id}")
        item = result_by_id[reference.task_id]
        if item.result.status == "error":
            raise InvalidEvidenceInputError(f"Evidence task failed: {reference.task_id}")
        if result_path_type(catalog.output_model(task.tool_name), reference.source_path) is None:
            raise InvalidEvidenceInputError(f"Evidence path is not a public result field: {reference.task_id}")
        try:
            value = resolve_result_path(item.result.data, reference.source_path)
        except ResultPathError as exc:
            raise InvalidEvidenceInputError(f"Evidence value is unavailable: {reference.task_id}") from exc
        resolved.append({"task_id": reference.task_id, "source_path": reference.source_path, "value": value})
    return resolved
