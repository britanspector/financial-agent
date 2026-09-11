"""Structured planning orchestration, independent of any concrete model provider."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from financial_agent.planner.models import StructuredPlan
from financial_agent.planner.prompt import build_planner_messages
from financial_agent.planner.providers import PlannerProvider
from financial_agent.schemas import Schema, UserQuery


class ToolCatalog(Protocol):
    def describe(self) -> list[dict]: ...
    def input_model(self, name: str) -> type[Schema] | None: ...


class StructuredPlanner:
    def __init__(self, provider: PlannerProvider, catalog: ToolCatalog) -> None:
        self._provider = provider
        self._catalog = catalog

    def plan(self, request: UserQuery) -> StructuredPlan:
        response_schema = _tool_aware_response_schema(self._catalog.describe())
        payload = self._provider.generate(
            build_planner_messages(request, self._catalog.describe()),
            response_schema=response_schema,
        )
        return StructuredPlan.model_validate(payload)


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
        },
        "required": ["task_id", "tool_name", "arguments", "dependencies"],
    }
    branches = []
    for tool in tools:
        branch = deepcopy(task_base)
        branch["properties"]["tool_name"] = {"const": tool["name"]}
        branch["properties"]["arguments"] = _strictify_object_schema(tool["input_schema"])
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
