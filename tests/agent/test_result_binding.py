from __future__ import annotations

from threading import Barrier, Lock
from uuid import uuid4

from financial_agent.agent.graph import run_execution_graph
from financial_agent.agent.models import ResultBinding
from financial_agent.agent.orchestrator import run_planner_execution
from financial_agent.agent.retry import RetryPolicy
from financial_agent.market_data.models import MarketSnapshot, MarketSnapshotInput
from financial_agent.planner.models import PlannedTask, StructuredPlan
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.tools.registry import ToolRegistry, ToolSpec
from financial_agent.user_data.runtime import register_user_tools


class NumberInput(Schema):
    value: int


class NumberOutput(Schema):
    value: int


class ListInput(Schema):
    values: list[int]


class ListOutput(Schema):
    values: list[int]


class BindingService:
    def __init__(self, *, fail: set[str] | None = None, barrier: Barrier | None = None):
        self.fail, self.barrier, self.calls = fail or set(), barrier, []
        self.active = self.max_active = 0
        self.lock = Lock()

    def execute(self, spec, arguments, *, context, request_id=None):
        del context
        with self.lock:
            self.calls.append(spec.name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            params = spec.input_model.model_validate(arguments)
            if self.barrier:
                self.barrier.wait(timeout=5)
            if spec.name in self.fail:
                return ToolResult(status="error", data=None, source="test", latency=0,
                    error=ToolError(code="FAIL", message="FAIL", http_status=503, retryable=False), request_id=uuid4())
            data = ({"values": [params.value]} if spec.name == "list_source" else {"values": params.values}) \
                if spec.name in {"list_source", "list_sink"} else {"value": params.value}
            return ToolResult(status="success", data=data, source="test", latency=0, error=None, request_id=uuid4())
        finally:
            with self.lock:
                self.active -= 1


def catalog(service=None):
    registry = ToolRegistry(service or BindingService())
    registry.register(ToolSpec("first", "", NumberInput, NumberOutput, None, "first"))
    registry.register(ToolSpec("second", "", NumberInput, NumberOutput, None, "second"))
    registry.register(ToolSpec("third", "", NumberInput, NumberOutput, None, "third"))
    registry.register(ToolSpec("list_source", "", NumberInput, ListOutput, None, "list_source"))
    registry.register(ToolSpec("list_sink", "", ListInput, ListOutput, None, "list_sink"))
    return registry


def planned(task_id, tool, arguments=None, bindings=None, dependencies=None):
    return PlannedTask(task_id=task_id, tool_name=tool, arguments=arguments or {},
        bindings=bindings or [], dependencies=dependencies or [])


def binding(target, source, *path):
    return ResultBinding(target_parameter=target, source_task_id=source, source_path=list(path))


def test_binding_compiles_dependency_and_resolves_multilevel_values():
    registry = catalog()
    plan = StructuredPlan(decision="execute", tasks=[
        planned("t1", "first", {"value": 7}),
        planned("t2", "second", bindings=[binding("value", "t1", "value")]),
        planned("t3", "third", bindings=[binding("value", "t2", "value")]),
    ])
    result = PlanValidator(registry).validate(plan)

    assert result.valid
    assert [task.dependencies for task in result.tasks] == [[], ["t1"], ["t2"]]
    state = run_execution_graph(UserQuery(query="chain"), result.tasks, registry)
    assert [item.result.data for item in state.final_output.task_results] == [{"value": 7}] * 3


def test_list_binding_and_invalid_binding_classes_are_checked():
    registry = catalog()
    valid = PlanValidator(registry).validate(StructuredPlan(decision="execute", tasks=[
        planned("src", "list_source", {"value": 1}),
        planned("sink", "list_sink", bindings=[binding("values", "src", "values")]),
    ]))
    assert valid.valid
    state = run_execution_graph(UserQuery(query="list"), valid.tasks, registry)
    assert state.final_output.task_results[1].result.data == {"values": [1]}
    invalid_cases = [
        (binding("value", "missing", "value"), "INVALID_BINDING_TASK"),
        (binding("value", "t1", "missing"), "INVALID_BINDING_FIELD"),
        (binding("value", "t1", "status"), "INVALID_BINDING_FIELD"),
        (binding("value", "t1", "values"), "BINDING_TYPE_MISMATCH"),
    ]
    for candidate, code in invalid_cases:
        result = PlanValidator(registry).validate(StructuredPlan(decision="execute", tasks=[
            planned("t1", "list_source", {"value": 1}), planned("t2", "second", bindings=[candidate]),
        ]))
        assert code in [issue.code for issue in result.issues]
    cycle = PlanValidator(registry).validate(StructuredPlan(decision="execute", tasks=[
        planned("left", "first", bindings=[binding("value", "right", "value")]),
        planned("right", "second", bindings=[binding("value", "left", "value")]),
    ]))
    assert "DEPENDENCY_CYCLE" in [issue.code for issue in cycle.issues]


def test_failed_upstream_binding_blocks_downstream_and_independent_branch_runs_in_parallel():
    service = BindingService(fail={"first"}, barrier=Barrier(2))
    registry = catalog(service)
    compiled = PlanValidator(registry).validate(StructuredPlan(decision="execute", tasks=[
        planned("root", "first", {"value": 1}),
        planned("child", "second", bindings=[binding("value", "root", "value")]),
        planned("free", "third", {"value": 3}),
    ])).tasks
    state = run_execution_graph(UserQuery(query="failure"), compiled, registry, max_concurrency=2)
    assert service.max_active == 2
    assert service.calls == ["first", "third"]
    assert state.final_output.task_results[1].result.error.code == "DEPENDENCY_FAILED"


class RetryBindingService(BindingService):
    def __init__(self):
        super().__init__()
        self.second_arguments = []
        self.second_request_ids = []

    def execute(self, spec, arguments, *, context, request_id=None):
        if spec.name == "second":
            self.calls.append(spec.name)
            self.second_arguments.append(dict(arguments))
            self.second_request_ids.append(request_id)
            if len(self.second_arguments) < 3:
                return ToolResult(
                    status="error", data=None, source="test", latency=0,
                    error=ToolError(
                        code="TEMPORARY_FAILURE", message="temporary", http_status=503, retryable=True,
                    ),
                    request_id=request_id,
                )
            return ToolResult(
                status="success", data={"value": arguments["value"]}, source="test", latency=0,
                error=None, request_id=request_id,
            )
        return super().execute(spec, arguments, context=context, request_id=request_id)


def test_retry_reuses_resolved_binding_without_reexecuting_upstream():
    service = RetryBindingService()
    registry = catalog(service)
    compiled = PlanValidator(registry).validate(StructuredPlan(decision="execute", tasks=[
        planned("source", "first", {"value": 7}),
        planned("sink", "second", bindings=[binding("value", "source", "value")]),
    ])).tasks

    state = run_execution_graph(
        UserQuery(query="stable binding"), compiled, registry, sleeper=lambda _: None,
    )

    assert service.calls.count("first") == 1
    assert service.calls.count("second") == 3
    assert service.second_arguments == [{"value": 7}] * 3
    assert len(set(service.second_request_ids)) == 3
    assert state.final_output.task_results[1].retry_count == 2
    assert state.final_output.iteration_count == 3


class E2EService:
    def execute(self, spec, arguments, *, context, request_id=None):
        del context, request_id
        if spec.name == "get_portfolio_positions":
            return ToolResult(status="success", source="synthetic", latency=0, error=None, request_id=uuid4(), data={
                "user_id": arguments["user_id"], "snapshot_date": "2026-09-10", "currency": "CNY", "industries": [],
                "stocks": [{"user_id": arguments["user_id"], "snapshot_date": "2026-09-10", "stock_code": "600519.SH", "industry_code": "food", "quantity": "1", "market_price": "1", "market_value": "1", "cost_price": "1", "cost_value": "1", "weight": 1, "profit_loss": "0", "profit_loss_pct": 0, "holding_days": 1, "last_trade_date": "2026-09-10"}],
            })
        return ToolResult(status="success", source="market", latency=0, error=None, request_id=uuid4(), data={"symbol": arguments["symbol"], "price": "123"})


class FixedPlanner:
    def generate(self, messages, *, response_schema):
        del messages, response_schema
        return {"decision": "execute", "tasks": [
            {"task_id": "positions", "tool_name": "get_portfolio_positions", "arguments": {"user_id": "syn-user-0001"}, "dependencies": [], "bindings": []},
            {"task_id": "quote", "tool_name": "get_market_snapshot", "arguments": {"symbol": None}, "dependencies": [], "bindings": [{"target_parameter": "symbol", "source_task_id": "positions", "source_path": ["stocks", 0, "stock_code"]}]},
        ]}


def test_unified_e2e_queries_largest_position_latest_quote():
    service = E2EService()
    user = register_user_tools(service)
    market = ToolRegistry(service)
    market.register(ToolSpec("get_market_snapshot", "", MarketSnapshotInput, MarketSnapshot, None, "snapshot"))
    from financial_agent.tools.composite import merge_registries
    registry = merge_registries(user, market)
    planner = StructuredPlanner(FixedPlanner(), registry)
    final = run_planner_execution(
        UserQuery(query="查询 syn-user-0001 最大仓位股票的最新行情"),
        planner,
        PlanValidator(registry),
        registry,
        retry_policy=RetryPolicy(max_retry=0),
    )
    assert final.status == "success"
    assert final.task_results[1].result.data["symbol"] == "600519.SH"
    assert all(item.max_retry == 0 for item in final.task_results)
