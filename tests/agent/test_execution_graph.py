from __future__ import annotations

from threading import Barrier, Lock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_agent.agent.graph import run_execution_graph
from financial_agent.agent.models import Task
from financial_agent.knowledge.runtime import register_rag_tools
from financial_agent.market_data.runtime import register_market_tools
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.composite import merge_registries
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.tools.registry import ToolRegistry, ToolSpec
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.runtime import register_user_tools


class ValueInput(Schema):
    value: int


class ValueOutput(Schema):
    value: int


class RecordingService:
    def __init__(self, *, barrier: Barrier | None = None, source: str = "test"):
        self.barrier = barrier
        self.source = source
        self.calls: list[str] = []
        self.max_active = 0
        self._active = 0
        self._lock = Lock()

    def execute(self, spec, arguments, *, context, request_id=None):
        del context
        with self._lock:
            self.calls.append(spec.name if spec else "unknown")
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            try:
                parameters = spec.input_model.model_validate(arguments)
            except ValidationError:
                return self._error("INVALID_ARGUMENT", retryable=False, request_id=request_id)
            if self.barrier is not None:
                self.barrier.wait(timeout=5)
            if spec.operation == "fail":
                return self._error("TEMPORARY_FAILURE", retryable=True, request_id=request_id)
            return ToolResult(
                status="success",
                data={"value": parameters.value},
                source=self.source,
                latency=0,
                error=None,
                request_id=request_id or uuid4(),
            )
        finally:
            with self._lock:
                self._active -= 1

    def _error(self, code, *, retryable, request_id):
        return ToolResult(
            status="error",
            data=None,
            source=self.source,
            latency=0,
            error=ToolError(code=code, message=code, http_status=503 if retryable else 422, retryable=retryable),
            request_id=request_id or uuid4(),
        )


def registry_with(*operations, service=None):
    service = service or RecordingService()
    registry = ToolRegistry(service)
    for operation in operations:
        registry.register(ToolSpec(operation, operation, ValueInput, ValueOutput, None, operation))
    return registry, service


def task(task_id, tool_name="ok", value=1, dependencies=None):
    return Task(
        task_id=task_id,
        tool_name=tool_name,
        arguments={"value": value},
        dependencies=dependencies or [],
    )


def test_single_task_executes_and_builds_final_result():
    registry, service = registry_with("ok")
    state = run_execution_graph(UserQuery(query="single"), [task("one")], registry)

    assert service.calls == ["ok"]
    assert state.status == "completed"
    assert state.final_output.status == "success"
    assert state.final_output.task_results[0].result.data == {"value": 1}


def test_empty_task_list_returns_successful_empty_result():
    registry, service = registry_with("ok")

    state = run_execution_graph(UserQuery(query="nothing to execute"), [], registry)

    assert service.calls == []
    assert state.final_output.status == "success"
    assert state.final_output.task_results == []
    assert state.final_output.errors == []


def test_dependency_free_tasks_execute_in_parallel():
    service = RecordingService(barrier=Barrier(2))
    registry, _ = registry_with("ok", service=service)

    state = run_execution_graph(
        UserQuery(query="parallel"),
        [task("one"), task("two", value=2)],
        registry,
        max_concurrency=2,
    )

    assert service.max_active == 2
    assert state.final_output.status == "success"
    assert [item.task_id for item in state.final_output.task_results] == ["one", "two"]


def test_dependencies_execute_in_topological_waves():
    registry, service = registry_with("first", "second", "third")
    tasks = [
        task("one", "first"),
        task("two", "second", dependencies=["one"]),
        task("three", "third", dependencies=["two"]),
    ]

    state = run_execution_graph(UserQuery(query="dependency"), tasks, registry)

    assert service.calls == ["first", "second", "third"]
    assert state.final_output.status == "success"


def test_tool_failure_enters_pool_and_blocks_dependents():
    registry, service = registry_with("fail", "ok")
    tasks = [task("root", "fail"), task("child", "ok", dependencies=["root"])]

    state = run_execution_graph(UserQuery(query="failure"), tasks, registry)

    assert service.calls == ["fail"]
    assert [item.result.error.code for item in state.tool_results] == [
        "TEMPORARY_FAILURE", "DEPENDENCY_FAILED",
    ]
    assert state.tool_results[0].result.error.retryable is True
    assert [error.reason for error in state.errors] == [None, "dependency_failed"]
    assert state.final_output.status == "failed"


def test_unknown_tool_and_invalid_arguments_are_results_not_exceptions():
    registry, _ = registry_with("ok")
    composite = merge_registries(registry)
    tasks = [
        task("unknown", "does_not_exist"),
        Task(task_id="invalid", tool_name="ok", arguments={"value": "bad"}, dependencies=[]),
    ]

    state = run_execution_graph(UserQuery(query="errors"), tasks, composite)

    assert [item.result.error.code for item in state.final_output.task_results] == [
        "UNKNOWN_TOOL", "INVALID_ARGUMENT",
    ]


def test_result_pool_aggregates_incremental_results_once_across_rounds():
    registry, _ = registry_with("ok", "fail")
    tasks = [
        task("one"),
        task("two", "ok", dependencies=["one"]),
        task("three", "fail", dependencies=["two"]),
        task("four", "ok", dependencies=["three"]),
    ]

    state = run_execution_graph(UserQuery(query="pool"), tasks, registry)

    assert [item.task_id for item in state.tool_results] == ["one", "two", "three", "four"]
    assert [error.task_id for error in state.errors] == ["three", "four"]
    assert len({item.task_id for item in state.tool_results}) == len(state.tool_results)
    assert state.final_output.status == "partial"


@pytest.mark.parametrize(
    ("tasks", "reason"),
    [
        ([task("one", dependencies=["absent"])], "missing_dependency"),
        ([task("one", dependencies=["two"]), task("two", dependencies=["one"])], "cycle_or_deadlock"),
    ],
)
def test_unresolved_dependencies_distinguish_reason(tasks, reason):
    registry, service = registry_with("ok")

    state = run_execution_graph(UserQuery(query="unresolved"), tasks, registry)

    assert service.calls == []
    assert all(error.code == "UNRESOLVED_DEPENDENCY" for error in state.errors)
    assert all(error.reason == reason for error in state.errors)
    assert all(item.result.error.message.startswith(reason) for item in state.tool_results)


def test_composite_exposes_all_nine_existing_tools_and_preserves_routing():
    user = RecordingService(source="user")
    market = RecordingService(source="market")
    rag = RecordingService(source="rag")
    composite = merge_registries(
        register_user_tools(user),
        register_market_tools(market),
        register_rag_tools(rag),
    )

    assert [item["name"] for item in composite.describe()] == [
        "get_customer_context", "get_margin_account", "get_portfolio_positions",
        "get_portfolio_analytics", "get_market_snapshot", "get_market_history",
        "search_research_reports", "search_regulatory_knowledge", "search_business_knowledge",
    ]
    result = composite.invoke("get_market_snapshot", {"value": 1}, context=CallContext())
    assert result.source == "market"
    assert market.calls == ["get_market_snapshot"]
    assert user.calls == [] and rag.calls == []


def test_composite_rejects_duplicate_tool_names():
    first, _ = registry_with("ok")
    second, _ = registry_with("ok")
    with pytest.raises(ValueError, match="Duplicate tool name"):
        merge_registries(first, second)
