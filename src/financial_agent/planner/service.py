"""Structured planning orchestration, independent of any concrete model provider."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.planner.models import ReplanOutput, StructuredPlan
from financial_agent.planner.prompt import build_planner_messages, build_replanner_messages
from financial_agent.verifier.models import VerificationResult
from financial_agent.planner.providers import PlannerProvider
from financial_agent.schemas import Schema, UserQuery


class ToolCatalog(Protocol):
    def describe(self) -> list[dict]: ...
    def input_model(self, name: str) -> type[Schema] | None: ...
    def output_model(self, name: str) -> type[BaseModel] | None: ...


class StructuredPlanner:
    def __init__(
        self,
        provider: PlannerProvider,
        catalog: ToolCatalog,
        *,
        context_manager: ContextManager | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._catalog = catalog
        self._context_manager = context_manager or ContextManager()
        self._context_policy = context_policy or ContextPolicy()

    def plan(self, request: UserQuery) -> StructuredPlan:
        contextual_request = self._context_manager.select(
            request, "planner", self._context_policy,
        ).request
        response_schema = _tool_aware_response_schema(self._catalog.describe())
        payload = self._provider.generate(
            build_planner_messages(contextual_request, self._catalog.describe()),
            response_schema=response_schema,
        )
        return StructuredPlan.model_validate(payload)

    def replan(
        self,
        request: UserQuery,
        previous_plan: list[Task],
        tool_results: list[TaskExecutionResult],
        feedback: VerificationResult,
    ) -> ReplanOutput:
        contextual_request = self._context_manager.select(
            request, "planner", self._context_policy,
        ).request
        tools = self._catalog.describe()
        payload = self._provider.generate(
            build_replanner_messages(contextual_request, tools, previous_plan, tool_results, feedback),
            response_schema=_replan_response_schema(tools),
        )
        output = ReplanOutput.model_validate(payload)
        task_ids = [task.task_id for task in output.tasks]
        forced = output.force_rerun_task_ids
        if len(forced) != len(set(forced)) or not set(forced).issubset(task_ids):
            raise ValueError("force_rerun_task_ids must be unique IDs in the replacement plan")
        return output


def _tool_aware_response_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Constrain task arguments to the selected public Tool input contract.

    A plain ``dict[str, Any]`` in the Pydantic model serializes as an open JSON
    object.  That left structured decoding unable to constrain argument names.
    The provider schema therefore uses one branch per public Tool, while Python
    keeps the generic task model needed by the execution boundary.
    """
    task_base = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "tool_name": {},
            "arguments": {},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "bindings": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "target_parameter": {"type": "string", "minLength": 1},
                        "source_task_id": {"type": "string", "minLength": 1},
                        "source_path": {"type": "array", "minItems": 1,
                                        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                    },
                    "required": ["target_parameter", "source_task_id", "source_path"],
                },
            },
        },
        "required": ["task_id", "tool_name", "arguments", "dependencies", "bindings"],
    }
    branches = []
    for tool in tools:
        branch = deepcopy(task_base)
        branch["properties"]["tool_name"] = {"const": tool["name"]}
        branch["properties"]["arguments"] = _binding_aware_object_schema(tool["input_schema"])
        branches.append(branch)

    execute = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"const": "execute"},
            "tasks": {"type": "array", "minItems": 1, "items": {"oneOf": branches}},
        },
        "required": ["decision", "tasks"],
    }
    non_execution = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"decision": {"const": decision}, "tasks": {"type": "array", "maxItems": 0}},
            "required": ["decision", "tasks"],
        }
        for decision in ("clarify", "no_tool")
    ]
    return {"oneOf": [execute, *non_execution]}


def _replan_response_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    execute = _tool_aware_response_schema(tools)["oneOf"][0]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tasks": execute["properties"]["tasks"],
            "force_rerun_task_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "required": ["tasks", "force_rerun_task_ids"],
    }


def _strictify_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make public input schemas compatible with strict structured decoding.

    Strict JSON-schema decoders require every declared object property to be
    present. Optional Tool fields remain semantically optional: the model emits
    their documented defaults (including ``null``/empty lists) and Pydantic
    validates them with the same public contract.
    """
    result = deepcopy(schema)
    if "properties" in result:
        result["additionalProperties"] = False
        result["required"] = list(result["properties"])
        result["properties"] = {
            name: _strictify_object_schema(value)
            for name, value in result["properties"].items()
        }
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _strictify_object_schema(result["items"])
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in result:
            result[combinator] = [_strictify_object_schema(value) for value in result[combinator]]
    return result


def _binding_aware_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep strict decoding while permitting ``null`` placeholders for bindings.

    Strict decoders require all object properties, including a required source
    parameter that will only exist after its upstream task runs.  A bound
    property may therefore be emitted as null; validation deliberately ignores
    that placeholder and runtime replaces it with the resolved public result.
    """
    result = _strictify_object_schema(schema)
    if "properties" in result:
        result["properties"] = {
            name: {"anyOf": [value, {"type": "null"}]}
            for name, value in result["properties"].items()
        }
    return result
