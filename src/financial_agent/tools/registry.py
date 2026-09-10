"""Small registry; schemas include tool arguments only, never credentials."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from financial_agent.schemas import Schema
from financial_agent.user_data.auth import CallContext, Scope
from financial_agent.tools.contracts import ToolResult

if TYPE_CHECKING:
    from financial_agent.user_data.service import UserDataService


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[Schema]
    output_model: type[Schema]
    scope: Scope | None
    operation: str


class ToolRegistry:
    def __init__(self, service: "UserDataService"):
        self._service = service
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError("Tool already registered")
        self._tools[spec.name] = spec

    def describe(self) -> list[dict]:
        return [
            {"name": spec.name, "description": spec.description,
             "input_schema": spec.input_model.model_json_schema()}
            for spec in self._tools.values()
        ]

    def invoke(self, name: str, arguments: object, *, context: CallContext, request_id: UUID | None = None) -> ToolResult:
        return self._service.execute(self._tools.get(name), arguments, context=context, request_id=request_id)
