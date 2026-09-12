"""Non-invasive routing across existing domain ToolRegistry instances."""

from __future__ import annotations

from time import perf_counter
from uuid import UUID, uuid4

from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.tools.registry import ToolRegistry
from financial_agent.user_data.auth import CallContext


class CompositeToolRegistry:
    """Route a tool call to its original registry without changing registry behavior."""

    def __init__(self, registries: tuple[ToolRegistry, ...]) -> None:
        self._registries = registries
        self._routes: dict[str, ToolRegistry] = {}
        self._descriptions: list[dict] = []
        for registry in registries:
            for description in registry.describe():
                name = description["name"]
                if name in self._routes:
                    raise ValueError(f"Duplicate tool name: {name}")
                self._routes[name] = registry
                self._descriptions.append(description)

    def describe(self) -> list[dict]:
        return list(self._descriptions)

    def input_model(self, name: str):
        registry = self._routes.get(name)
        return registry.input_model(name) if registry is not None else None

    def output_model(self, name: str):
        registry = self._routes.get(name)
        return registry.output_model(name) if registry is not None else None

    def invoke(
        self,
        name: str,
        arguments: object,
        *,
        context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult:
        registry = self._routes.get(name)
        if registry is not None:
            return registry.invoke(name, arguments, context=context, request_id=request_id)
        started = perf_counter()
        return ToolResult(
            status="error",
            data=None,
            source="tool_registry",
            latency=max(0.0, (perf_counter() - started) * 1_000),
            error=ToolError(
                code="UNKNOWN_TOOL",
                message="Unknown tool",
                http_status=404,
                retryable=False,
            ),
            request_id=request_id or uuid4(),
        )


def merge_registries(*registries: ToolRegistry) -> CompositeToolRegistry:
    if not registries:
        raise ValueError("At least one registry is required")
    return CompositeToolRegistry(tuple(registries))
