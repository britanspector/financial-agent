"""Deterministic structural validation before Phase 2 execution."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pydantic import ValidationError

from financial_agent.agent.models import Task
from financial_agent.planner.models import (
    PlanValidationIssue,
    PlanValidationResult,
    StructuredPlan,
)
from financial_agent.planner.service import ToolCatalog


_RESULT_REFERENCE = re.compile(r"\$[A-Za-z0-9_-]+\.result(?:\.|$)")


class PlanValidator:
    def __init__(self, catalog: ToolCatalog, *, max_tasks: int = 12) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        self._catalog = catalog
        self._max_tasks = max_tasks

    def validate(self, plan: StructuredPlan) -> PlanValidationResult:
        issues: list[PlanValidationIssue] = []
        normalized: dict[int, dict[str, Any]] = {}
        if plan.decision == "execute" and not plan.tasks:
            issues.append(self._issue("EMPTY_PLAN", "Plan must contain at least one task"))
        if plan.decision != "execute" and plan.tasks:
            issues.append(self._issue(
                "NON_EXECUTION_HAS_TASKS", "clarify and no_tool plans must not contain tasks",
            ))
        if len(plan.tasks) > self._max_tasks:
            issues.append(self._issue("TOO_MANY_TASKS", f"Plan exceeds {self._max_tasks} tasks"))

        id_counts = Counter(task.task_id for task in plan.tasks)
        for task_id, count in id_counts.items():
            if count > 1:
                issues.append(self._issue("DUPLICATE_TASK_ID", "Task ID must be unique", task_id))

        known_ids = set(id_counts)
        for index, task in enumerate(plan.tasks):
            model = self._catalog.input_model(task.tool_name)
            if model is None:
                issues.append(self._issue("UNKNOWN_TOOL", f"Unknown tool: {task.tool_name}", task.task_id))
            else:
                try:
                    value = model.model_validate(task.arguments)
                    normalized[index] = value.model_dump(mode="json")
                except ValidationError as exc:
                    detail = "; ".join(error["msg"] for error in exc.errors(include_input=False))
                    issues.append(self._issue("INVALID_ARGUMENTS", detail, task.task_id))
            if _contains_result_reference(task.arguments):
                issues.append(self._issue(
                    "DYNAMIC_RESULT_REFERENCE", "Result binding is not supported in Phase 3.1", task.task_id,
                ))
            if len(task.dependencies) != len(set(task.dependencies)):
                issues.append(self._issue("DUPLICATE_DEPENDENCY", "Dependencies must be unique", task.task_id))
            if task.task_id in task.dependencies:
                issues.append(self._issue("SELF_DEPENDENCY", "Task cannot depend on itself", task.task_id))
            for dependency in dict.fromkeys(task.dependencies):
                if dependency not in known_ids:
                    issues.append(self._issue(
                        "MISSING_DEPENDENCY", f"Missing dependency: {dependency}", task.task_id,
                    ))

        if _has_cycle(plan, known_ids):
            issues.append(self._issue("DEPENDENCY_CYCLE", "Task dependencies contain a cycle"))

        if issues:
            return PlanValidationResult(valid=False, decision=plan.decision, issues=issues)
        if plan.decision != "execute":
            return PlanValidationResult(valid=True, decision=plan.decision)
        tasks = [
            Task(
                task_id=task.task_id,
                tool_name=task.tool_name,
                arguments=normalized[index],
                dependencies=task.dependencies,
            )
            for index, task in enumerate(plan.tasks)
        ]
        return PlanValidationResult(valid=True, decision=plan.decision, tasks=tasks)

    def normalize_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Return Tool-schema defaults in JSON form, or ``None`` if invalid.

        Evaluation uses this to compare semantically equivalent optional fields;
        it does not alter Planner output or execution behavior.
        """
        model = self._catalog.input_model(tool_name)
        if model is None:
            return None
        try:
            return model.model_validate(arguments).model_dump(mode="json")
        except ValidationError:
            return None

    @staticmethod
    def _issue(code: str, message: str, task_id: str | None = None) -> PlanValidationIssue:
        return PlanValidationIssue(code=code, message=message, task_id=task_id)


def _contains_result_reference(value: Any) -> bool:
    if isinstance(value, str):
        return _RESULT_REFERENCE.search(value) is not None
    if isinstance(value, dict):
        return any(_contains_result_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_result_reference(item) for item in value)
    return False


def _has_cycle(plan: StructuredPlan, known_ids: set[str]) -> bool:
    graph: dict[str, set[str]] = {task_id: set() for task_id in known_ids}
    for task in plan.tasks:
        graph[task.task_id].update(
            dependency for dependency in task.dependencies
            if dependency in known_ids and dependency != task.task_id
        )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        if any(visit(dependency) for dependency in graph[task_id]):
            return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    return any(visit(task_id) for task_id in graph if task_id not in visited)
