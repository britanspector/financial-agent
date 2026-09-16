"""Deterministic, offline end-to-end evaluation for the complete Agent Loop."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from financial_agent.agent.loop import AgentLoopResult, LoopPolicy, run_agent_loop
from financial_agent.agent.retry import RetryPolicy
from financial_agent.answering.service import AnswerWriter
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.demo_faults import FaultSequence
from financial_agent.knowledge.models import (
    AnnouncementPolicyMetadata,
    BusinessSearchInput,
    Evidence,
    FAQMetadata,
    RegulatorySearchInput,
    ResearchReportMetadata,
    ResearchSearchInput,
)
from financial_agent.knowledge.runtime import register_rag_tools
from financial_agent.knowledge.service import KnowledgeRetrievalService
from financial_agent.market_data.models import MarketBar, MarketHistory, MarketSnapshot
from financial_agent.market_data.runtime import register_market_tools
from financial_agent.market_data.service import MarketDataService
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.composite import merge_registries
from financial_agent.user_data.auth import CallContext, Credential, CredentialStore
from financial_agent.user_data.models import MarginAccountInput
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.service import UserDataService
from financial_agent.verifier.service import StructuredVerifier


class ReplayCall(Schema):
    output: dict[str, Any]
    required_prompt_fragments: list[str] = Field(default_factory=list)


class E2EScenario(Schema):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)
    planner_calls: list[ReplayCall] = Field(min_length=1)
    answer_calls: list[ReplayCall] = Field(min_length=1)
    verifier_calls: list[ReplayCall] = Field(min_length=1)
    user_faults: list[Literal["timeout", "429", "503", "success"]] = Field(default_factory=list)
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    loop_policy: LoopPolicy = Field(default_factory=LoopPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class LogicalTool(Schema):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ExpectedEvidence(LogicalTool):
    source_path: list[str | int] = Field(min_length=1)


class E2EExpectation(Schema):
    case_id: str = Field(min_length=1)
    solvable: bool = True
    status: Literal["completed", "limit_exhausted", "clarify", "no_tool"]
    stop_reason: str
    expected_tools: list[LogicalTool]
    required_text: list[str] = Field(default_factory=list)
    forbidden_text: list[str] = Field(default_factory=list)
    required_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    expected_tool_attempts: int = Field(ge=0)
    expected_loop_iterations: int = Field(ge=0)
    expected_rewrite_count: int = Field(default=0, ge=0)
    expected_replan_count: int = Field(default=0, ge=0)
    expected_tool_selection_correct: bool = True
    expected_unnecessary_calls: int = Field(default=0, ge=0)
    required_retrieved_message_indexes: list[int] = Field(default_factory=list)


class ToolInvocation(Schema):
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["success", "empty", "error"]
    error_code: str | None = None


class ContextObservation(Schema):
    component: Literal["planner", "writer", "verifier"]
    retrieval_active: bool
    retrieved_message_indexes: list[int]


class PlanObservation(Schema):
    action: Literal["initial", "replan"]
    tools: list[LogicalTool]


class E2ECaseResult(Schema):
    case_id: str
    passed: bool
    failures: list[str]
    task_success: bool
    answer_correct: bool
    tool_selection_correct: bool
    unnecessary_tool_calls: int = Field(ge=0)
    actual_logical_tools: list[LogicalTool]
    tool_invocations: list[ToolInvocation]
    context_observations: list[ContextObservation]
    plan_observations: list[PlanObservation]
    result: AgentLoopResult


class AgentE2EMetrics(Schema):
    task_success_rate: float = Field(ge=0, le=1)
    final_answer_correctness: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)
    replan_recovery_rate: float = Field(ge=0, le=1)
    average_tool_attempts: float = Field(ge=0)
    loop_iterations: float = Field(ge=0)


class AgentE2EEvaluationReport(Schema):
    case_count: int = Field(ge=1)
    passed_count: int = Field(ge=0)
    all_passed: bool
    metrics: AgentE2EMetrics
    cases: list[E2ECaseResult]


def load_e2e_eval_set(
    scenarios_path: Path, expectations_path: Path,
) -> tuple[list[E2EScenario], list[E2EExpectation]]:
    scenarios = _load_jsonl(scenarios_path, E2EScenario)
    expectations = _load_jsonl(expectations_path, E2EExpectation)
    scenario_ids = [item.case_id for item in scenarios]
    expectation_ids = [item.case_id for item in expectations]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("E2E scenario IDs must be unique")
    if len(expectation_ids) != len(set(expectation_ids)):
        raise ValueError("E2E expectation IDs must be unique")
    if scenario_ids != expectation_ids:
        raise ValueError("E2E scenarios and expectations must have identical ordered case IDs")
    return scenarios, expectations


def evaluate_agent_e2e(
    scenarios: list[E2EScenario], expectations: list[E2EExpectation],
) -> AgentE2EEvaluationReport:
    if not scenarios or [item.case_id for item in scenarios] != [item.case_id for item in expectations]:
        raise ValueError("E2E scenarios and expectations must be non-empty and aligned")
    observations = [_run_case(case, expected) for case, expected in zip(scenarios, expectations)]
    solvable = [item for item, expected in zip(observations, expectations) if expected.solvable]
    replan_targets = [
        item for item, expected in zip(observations, expectations)
        if expected.solvable and expected.expected_replan_count > 0
    ]
    logical_count = sum(len(item.actual_logical_tools) for item in observations)
    metrics = AgentE2EMetrics(
        task_success_rate=_rate(sum(item.task_success for item in solvable), len(solvable)),
        final_answer_correctness=_rate(sum(item.answer_correct for item in solvable), len(solvable)),
        tool_selection_accuracy=_rate(
            sum(item.tool_selection_correct for item in observations), len(observations),
        ),
        unnecessary_tool_call_rate=_rate(
            sum(item.unnecessary_tool_calls for item in observations), logical_count,
        ),
        replan_recovery_rate=_rate(
            sum(item.task_success for item in replan_targets), len(replan_targets),
        ),
        average_tool_attempts=sum(len(item.tool_invocations) for item in observations) / len(observations),
        loop_iterations=sum(item.result.iteration_count for item in observations) / len(observations),
    )
    return AgentE2EEvaluationReport(
        case_count=len(observations),
        passed_count=sum(item.passed for item in observations),
        all_passed=all(item.passed for item in observations),
        metrics=metrics,
        cases=observations,
    )


def _run_case(case: E2EScenario, expected: E2EExpectation) -> E2ECaseResult:
    registry, call_context = _fixture_registry(case.user_faults)
    recording_registry = _RecordingRegistry(registry)
    context_manager = _RecordingContextManager(ContextManager())
    planner = _RecordingPlanner(StructuredPlanner(
        _ReplayProvider(case.planner_calls), recording_registry,
        context_manager=context_manager, context_policy=case.context_policy,
    ))
    writer = AnswerWriter(
        _ReplayProvider(case.answer_calls), recording_registry,
        context_manager=context_manager, context_policy=case.context_policy,
    )
    verifier = StructuredVerifier(
        _ReplayProvider(case.verifier_calls), recording_registry,
        context_manager=context_manager, context_policy=case.context_policy,
    )
    result = run_agent_loop(
        UserQuery(query=case.query, history=case.history),
        planner,
        PlanValidator(recording_registry),
        recording_registry,
        writer,
        verifier,
        policy=case.loop_policy,
        retry_policy=case.retry_policy,
        context=call_context,
        max_concurrency=1,
        sleeper=lambda _: None,
    )
    actual_tools = recording_registry.logical_tools()
    normalized_expected = [recording_registry.normalize(item) for item in expected.expected_tools]
    actual_counter = Counter(_tool_key(item) for item in actual_tools)
    expected_counter = Counter(_tool_key(item) for item in normalized_expected)
    selection_correct = actual_counter == expected_counter
    unnecessary = sum((actual_counter - expected_counter).values())
    answer_correct = _answer_correct(result, expected, recording_registry)
    final_results_ok = all(item.result.status != "error" for item in result.task_results)
    task_success = result.status == "completed" and answer_correct and final_results_ok
    retrieved = {
        index
        for observation in context_manager.observations
        for index in observation.retrieved_message_indexes
        if observation.retrieval_active
    }
    comparisons = {
        "status": result.status == expected.status,
        "stop_reason": result.stop_reason == expected.stop_reason,
        "tool_attempts": result.total_tool_attempts == expected.expected_tool_attempts,
        "loop_iterations": result.iteration_count == expected.expected_loop_iterations,
        "rewrite_count": result.rewrite_count == expected.expected_rewrite_count,
        "replan_count": result.replan_count == expected.expected_replan_count,
        "answer": answer_correct if expected.solvable else True,
        "tool_selection_behavior": selection_correct == expected.expected_tool_selection_correct,
        "unnecessary_call_behavior": unnecessary == expected.expected_unnecessary_calls,
        "context_retrieval": set(expected.required_retrieved_message_indexes).issubset(retrieved),
    }
    failures = [name for name, passed in comparisons.items() if not passed]
    return E2ECaseResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        task_success=task_success,
        answer_correct=answer_correct,
        tool_selection_correct=selection_correct,
        unnecessary_tool_calls=unnecessary,
        actual_logical_tools=actual_tools,
        tool_invocations=recording_registry.invocations,
        context_observations=context_manager.observations,
        plan_observations=planner.observations,
        result=result,
    )


def _answer_correct(result, expected, registry) -> bool:
    if result.answer is None or result.draft is None:
        return False
    folded = result.answer.casefold()
    if any(text.casefold() not in folded for text in expected.required_text):
        return False
    if any(text.casefold() in folded for text in expected.forbidden_text):
        return False
    tasks = {task.task_id: task for task in result.plan}
    actual = set()
    for reference in result.draft.evidence:
        task = tasks.get(reference.task_id)
        if task is None:
            continue
        try:
            candidates = [registry.normalize(LogicalTool(
                tool_name=task.tool_name, arguments=task.arguments,
            ))]
        except ValueError:
            candidates = [
                item for item in registry.logical_tools() if item.tool_name == task.tool_name
            ]
        actual.update((_tool_key(item), tuple(reference.source_path)) for item in candidates)
    required = {
        (_tool_key(registry.normalize(item)), tuple(item.source_path))
        for item in expected.required_evidence
    }
    return required.issubset(actual)


class _ReplayProvider:
    def __init__(self, calls: list[ReplayCall]) -> None:
        self._calls = list(calls)

    def generate(self, messages, *, response_schema):
        del response_schema
        if not self._calls:
            raise AssertionError("Unexpected replay provider call")
        call = self._calls.pop(0)
        prompt = json.dumps(messages, ensure_ascii=False)
        missing = [item for item in call.required_prompt_fragments if item not in prompt]
        if missing:
            raise AssertionError(f"Required prompt fragments missing: {missing}")
        return call.output


class _RecordingRegistry:
    def __init__(self, registry) -> None:
        self._registry = registry
        self.invocations: list[ToolInvocation] = []

    def describe(self):
        return self._registry.describe()

    def input_model(self, name):
        return self._registry.input_model(name)

    def output_model(self, name):
        return self._registry.output_model(name)

    def invoke(self, name, arguments, *, context, request_id=None):
        normalized = self.normalize(LogicalTool(tool_name=name, arguments=arguments)).arguments
        result = self._registry.invoke(name, arguments, context=context, request_id=request_id)
        self.invocations.append(ToolInvocation(
            tool_name=name,
            arguments=normalized,
            status=result.status,
            error_code=result.error.code if result.error else None,
        ))
        return result

    def normalize(self, item: LogicalTool) -> LogicalTool:
        model = self.input_model(item.tool_name)
        if model is None:
            raise ValueError(f"Unknown E2E Tool: {item.tool_name}")
        arguments = model.model_validate(item.arguments).model_dump(mode="json")
        return LogicalTool(tool_name=item.tool_name, arguments=arguments)

    def logical_tools(self) -> list[LogicalTool]:
        unique: dict[str, LogicalTool] = {}
        for call in self.invocations:
            item = LogicalTool(tool_name=call.tool_name, arguments=call.arguments)
            unique.setdefault(_tool_key(item), item)
        return list(unique.values())


class _RecordingPlanner:
    def __init__(self, planner: StructuredPlanner) -> None:
        self._planner = planner
        self.observations: list[PlanObservation] = []

    def plan(self, request):
        output = self._planner.plan(request)
        self._record("initial", output.tasks)
        return output

    def replan(self, request, previous_plan, tool_results, feedback):
        output = self._planner.replan(request, previous_plan, tool_results, feedback)
        self._record("replan", output.tasks)
        return output

    def _record(self, action, tasks) -> None:
        self.observations.append(PlanObservation(
            action=action,
            tools=[LogicalTool(tool_name=task.tool_name, arguments=task.arguments) for task in tasks],
        ))


class _RecordingContextManager:
    def __init__(self, manager: ContextManager) -> None:
        self._manager = manager
        self.observations: list[ContextObservation] = []

    def select(self, request, component, policy):
        selection = self._manager.select(request, component, policy)
        self.observations.append(ContextObservation(
            component=component,
            retrieval_active=selection.metrics.retrieval_active,
            retrieved_message_indexes=sorted({
                index for turn in selection.retrieved_history for index in turn.message_indexes
            }),
        ))
        return selection


class _MemoryAudit:
    def write(self, event) -> None:
        del event


class _FixtureRepository:
    def get_customer_context(self, user_id):
        return {
            "user_id": user_id, "name_alias": "客户甲", "age_band": "30-39", "risk_level": "R3",
            "region": "华东", "customer_tier": "gold", "created_at": "2026-01-01T00:00:00Z",
            "snapshot_date": "2026-09-15", "asset_bucket": "1m-5m", "investable_asset": "1500000",
            "total_asset": "1800000", "cash_asset": "300000", "risk_tolerance_score": 0.6,
            "investment_experience_years": 8, "investment_horizon": "medium",
            "liquidity_need_score": 0.3, "active_trading_days_90d": 20, "trade_enabled": True,
            "margin_enabled": True, "short_selling_enabled": False,
            "max_allowed_product_risk_level": "R3",
        }

    def get_margin_account(self, query: MarginAccountInput):
        return {
            "user_id": query.user_id,
            "info": {
                "user_id": query.user_id, "snapshot_date": "2026-09-15", "margin_account_id": "M-001",
                "credit_limit": "500000", "financing_balance": "100000",
                "securities_lending_balance": "0", "collateral_market_value": "1200000",
                "cash_balance": "200000", "available_credit": "400000",
                "maintenance_margin_ratio": 2.0, "warning_line": 1.4, "liquidation_line": 1.3,
                "credit_utilization": 0.2, "margin_call_flag": False,
                "forced_liquidation_flag": False,
            },
            "daily": [], "total": 0, "limit": query.limit, "offset": query.offset,
            "start_date": str(query.start_date) if query.start_date else None,
            "end_date": str(query.end_date) if query.end_date else None,
        }

    def get_portfolio_positions(self, user_id):
        return {
            "user_id": user_id, "snapshot_date": "2026-09-15", "currency": "CNY",
            "industries": [],
            "stocks": [{
                "user_id": user_id, "snapshot_date": "2026-09-15", "stock_code": "600519.SH",
                "industry_code": "FOOD", "quantity": "100", "market_price": "1500",
                "market_value": "150000", "cost_price": "1400", "cost_value": "140000",
                "weight": 0.1, "profit_loss": "10000", "profit_loss_pct": 0.071428,
                "holding_days": 120, "last_trade_date": "2026-09-01",
            }],
        }

    def get_portfolio_analytics(self, user_id):
        return {
            "user_id": user_id,
            "report": {
                "user_id": user_id, "report_date": "2026-09-15", "period_start": "2026-01-01",
                "period_end": "2026-09-15", "beginning_equity": "1000000", "period_net_pnl": "80000",
                "realized_pnl": "50000", "beginning_unrealized_pnl": "10000",
                "ending_unrealized_pnl": "45000", "financing_interest": "2000",
                "securities_lending_fee": "0", "transaction_fees": "2000", "other_fees": "1000",
                "portfolio_return": 0.08, "annualized_return": 0.12, "volatility": 0.18,
                "max_drawdown": -0.06, "sharpe_ratio": 0.9, "win_rate": 0.58,
                "turnover_rate": 1.2, "average_holding_days": 80, "profit_trade_ratio": 0.6,
                "margin_contribution": 0.01, "benchmark_excess_return": 0.03,
            },
            "factors": {
                "user_id": user_id, "snapshot_date": "2026-09-15", "market_beta": 0.9,
                "size_exposure": 0.1, "value_exposure": 0.2, "growth_exposure": 0.1,
                "momentum_exposure": 0.2, "quality_exposure": 0.4, "volatility_exposure": -0.1,
                "liquidity_exposure": 0.1, "leverage_exposure": 0.2, "concentration_exposure": 0.3,
            },
            "rank": {
                "user_id": user_id, "rank_date": "2026-09-15", "peer_group_key": "华东:1m-5m",
                "peer_group_size": 100, "period_return": 0.08, "benchmark_excess_return": 0.03,
                "return_rank": 20, "return_percentile": 0.8, "drawdown_rank": 15,
                "risk_adjusted_rank": 18,
            },
        }


class _FixtureMarketProvider:
    def get_snapshot(self, symbol):
        return MarketSnapshot(
            symbol=symbol, snapshot_kind="daily_close",
            as_of=datetime(2026, 9, 15, 15, tzinfo=timezone.utc), price=Decimal("1500"),
            open=Decimal("1490"), high=Decimal("1510"), low=Decimal("1480"),
            previous_close=Decimal("1485"), pct_change=Decimal("1.01"),
            volume=Decimal("1000000"), amount=Decimal("1500000000"), source="tushare",
        )

    def get_history(self, symbol, start_date, end_date):
        return MarketHistory(
            symbol=symbol, asset_type="stock", start_date=start_date, end_date=end_date,
            adjustment="none", source="tushare", bars=[MarketBar(
                trade_date=start_date, open=Decimal("100"), high=Decimal("105"),
                low=Decimal("99"), close=Decimal("103"), previous_close=Decimal("100"),
                pct_change=Decimal("3"), volume=Decimal("10000"), amount=Decimal("1030000"),
                source="tushare",
            )],
        )


class _FixtureKnowledgeRetriever:
    def search_research(self, request: ResearchSearchInput):
        company = request.companies[0] if request.companies else "示例公司"
        metadata = ResearchReportMetadata(
            document_id="research-1", source_type="research_report", title="示例研报",
            company=company, broker="示例证券", publish_date=date(2026, 9, 1), report_period="2026H1",
        )
        return [_evidence(metadata, "research-1", "盈利保持稳定")]

    def search_regulatory(self, request: RegulatorySearchInput):
        metadata = AnnouncementPolicyMetadata(
            document_id="policy-1", source_type="announcement_policy", title="适当性规则",
            issuer=request.issuer or "监管机构", publish_date=date(2026, 1, 1),
            effective_date=date(2026, 2, 1), status="active",
        )
        return [_evidence(metadata, "policy-1", "产品风险等级不得高于客户承受等级")]

    def search_business(self, request: BusinessSearchInput):
        metadata = FAQMetadata(
            document_id="faq-1", source_type="faq", title="两融说明",
            category=request.category or "margin", version="1.0", effective_date=date(2026, 1, 1),
        )
        return [_evidence(metadata, "faq-1", "融资余额会占用授信额度")]


def _evidence(metadata, chunk_id: str, content: str) -> Evidence:
    return Evidence(
        chunk_id=chunk_id, document_id=metadata.document_id, source_type=metadata.source_type,
        title=metadata.title, content=content, score=1.0, metadata=metadata,
    )


def _fixture_registry(user_faults):
    ephemeral_key = f"e2e-{uuid4().hex}"
    credentials = CredentialStore([Credential(
        api_key=ephemeral_key, principal_id="e2e",
        user_ids={"syn-user-0001", "syn-user-0002"},
        scopes={
            "read:customer_context", "read:margin_account", "read:portfolio_positions",
            "read:portfolio_analytics",
        },
    )])
    service = UserDataService(
        _FixtureRepository(), credentials, _MemoryAudit(),
        before_read=FaultSequence(user_faults) if user_faults else None,
    )
    registry = merge_registries(
        register_user_tools(service),
        register_market_tools(MarketDataService(_FixtureMarketProvider())),
        register_rag_tools(KnowledgeRetrievalService(_FixtureKnowledgeRetriever())),
    )
    return registry, CallContext(api_key=ephemeral_key)


def _load_jsonl(path: Path, model):
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid E2E eval entry at {path}:{line_number}") from exc
    if not values:
        raise ValueError(f"E2E eval file is empty: {path}")
    return values


def _tool_key(item: LogicalTool) -> str:
    return f"{item.tool_name}:{json.dumps(item.arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
