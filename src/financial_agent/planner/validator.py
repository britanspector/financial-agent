"""Deterministic plan and result-binding validation before Phase 2 execution."""

from __future__ import annotations

from collections import Counter
import re
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from financial_agent.agent.models import Task
from financial_agent.agent.result_path import result_path_type
from financial_agent.planner.models import PlanValidationIssue, PlanValidationResult, StructuredPlan
from financial_agent.planner.service import ToolCatalog

_LEGACY_RESULT_REFERENCE = re.compile(r"\$[A-Za-z0-9_-]+\.result(?:\.|$)")


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
            issues.append(self._issue("NON_EXECUTION_HAS_TASKS", "clarify and no_tool plans must not contain tasks"))
        if len(plan.tasks) > self._max_tasks:
            issues.append(self._issue("TOO_MANY_TASKS", f"Plan exceeds {self._max_tasks} tasks"))
        id_counts = Counter(task.task_id for task in plan.tasks)
        for task_id, count in id_counts.items():
            if count > 1:
                issues.append(self._issue("DUPLICATE_TASK_ID", "Task ID must be unique", task_id))
        known_ids = set(id_counts)
        tasks_by_id = {task.task_id: task for task in plan.tasks}
        for index, task in enumerate(plan.tasks):
            model = self._catalog.input_model(task.tool_name)
            bound_targets = {binding.target_parameter for binding in task.bindings}
            if model is None:
                issues.append(self._issue("UNKNOWN_TOOL", f"Unknown tool: {task.tool_name}", task.task_id))
            else:
                arguments, argument_issue = _validate_arguments(model, task.arguments, bound_targets)
                if argument_issue:
                    issues.append(self._issue("INVALID_ARGUMENTS", argument_issue, task.task_id))
                else:
                    normalized[index] = arguments
            if _contains_legacy_result_reference(task.arguments):
                issues.append(self._issue("DYNAMIC_RESULT_REFERENCE", "Use structured bindings instead of result-reference strings", task.task_id))
            if len(bound_targets) != len(task.bindings):
                issues.append(self._issue("DUPLICATE_BINDING_TARGET", "A parameter may have only one binding", task.task_id))
            for binding in task.bindings:
                source = tasks_by_id.get(binding.source_task_id)
                if source is None:
                    issues.append(self._issue("INVALID_BINDING_TASK", "Binding source task does not exist", task.task_id))
                    continue
                if binding.source_task_id == task.task_id:
                    issues.append(self._issue("SELF_BINDING", "Task cannot bind its own result", task.task_id))
                    continue
                if model is None:
                    continue
                target = model.model_fields.get(binding.target_parameter)
                if target is None:
                    issues.append(self._issue("INVALID_BINDING_FIELD", "Binding target is not a Tool parameter", task.task_id))
                    continue
                source_type = result_path_type(self._catalog.output_model(source.tool_name), binding.source_path)
                if source_type is None:
                    issues.append(self._issue("INVALID_BINDING_FIELD", "Binding source path is not a public result field", task.task_id))
                elif not _types_compatible(source_type, target.annotation):
                    issues.append(self._issue("BINDING_TYPE_MISMATCH", "Binding result type does not match target parameter", task.task_id))
            effective_dependencies = _effective_dependencies(task)
            if len(task.dependencies) != len(set(task.dependencies)):
                issues.append(self._issue("DUPLICATE_DEPENDENCY", "Dependencies must be unique", task.task_id))
            if task.task_id in effective_dependencies:
                issues.append(self._issue("SELF_DEPENDENCY", "Task cannot depend on itself", task.task_id))
            for dependency in effective_dependencies:
                if dependency not in known_ids:
                    issues.append(self._issue("MISSING_DEPENDENCY", f"Missing dependency: {dependency}", task.task_id))
        if _has_cycle(plan, known_ids):
            issues.append(self._issue("DEPENDENCY_CYCLE", "Task dependencies contain a cycle"))
        if issues:
            return PlanValidationResult(valid=False, decision=plan.decision, issues=issues)
        if plan.decision != "execute":
            return PlanValidationResult(valid=True, decision=plan.decision)
        return PlanValidationResult(valid=True, decision=plan.decision, tasks=[
            Task(task_id=task.task_id, tool_name=task.tool_name, arguments=normalized[index],
                 dependencies=_effective_dependencies(task), bindings=task.bindings)
            for index, task in enumerate(plan.tasks)
        ])

    def normalize_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
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


def _validate_arguments(model: type[BaseModel], arguments: dict[str, Any], bound: set[str]) -> tuple[dict[str, Any], str | None]:
    unknown = set(arguments) - set(model.model_fields)
    if unknown:
        return {}, f"Extra inputs are not permitted: {', '.join(sorted(unknown))}"
    missing = [name for name, field in model.model_fields.items() if field.is_required() and name not in arguments and name not in bound]
    if missing:
        return {}, f"Field required: {', '.join(missing)}"
    normalized = dict(arguments)
    try:
        for name, value in arguments.items():
            if name in bound:
                continue
            normalized[name] = TypeAdapter(model.model_fields[name].annotation).validate_python(value)
    except ValidationError as exc:
        return {}, "; ".join(error["msg"] for error in exc.errors(include_input=False))
    if bound:
        return normalized, None
    try:
        return model.model_validate(normalized).model_dump(mode="json"), None
    except ValidationError as exc:
        return {}, "; ".join(error["msg"] for error in exc.errors(include_input=False))


def _effective_dependencies(task) -> list[str]:
    return list(dict.fromkeys([*(binding.source_task_id for binding in task.bindings), *task.dependencies]))


def _contains_legacy_result_reference(value: Any) -> bool:
    if isinstance(value, str):
        return _LEGACY_RESULT_REFERENCE.search(value) is not None
    if isinstance(value, dict):
        return any(_contains_legacy_result_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_legacy_result_reference(item) for item in value)
    return False


def _types_compatible(source: Any, target: Any) -> bool:
    if source is Any or target is Any:
        return True
    source_origin, target_origin = get_origin(source), get_origin(target)
    if source_origin is list or target_origin is list:
        return source_origin is target_origin is list and _types_compatible(get_args(source)[0], get_args(target)[0])
    if target_origin in (UnionType, Union):
        return any(_types_compatible(source, option) for option in get_args(target) if option is not type(None))
    if source_origin in (UnionType, Union):
        return all(_types_compatible(option, target) for option in get_args(source) if option is not type(None))
    try:
        return issubclass(source, target)
    except TypeError:
        return source == target


def _has_cycle(plan: StructuredPlan, known_ids: set[str]) -> bool:
    graph = {task_id: set() for task_id in known_ids}
    for task in plan.tasks:
        graph[task.task_id].update(dep for dep in _effective_dependencies(task) if dep in known_ids and dep != task.task_id)
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        found = any(visit(dep) for dep in graph[task_id])
        visiting.remove(task_id)
        visited.add(task_id)
        return found
    return any(visit(task_id) for task_id in graph if task_id not in visited)
