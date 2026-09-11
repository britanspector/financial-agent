import pytest

from financial_agent.config import Settings
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import UserQuery

from .conftest import planner_catalog


@pytest.mark.live
def test_qwen37_flash_returns_valid_single_tool_plan():
    settings = Settings()
    if settings.qwen_api_key is None:
        pytest.skip("FINANCIAL_AGENT_QWEN_API_KEY is not configured")
    key = settings.qwen_api_key.get_secret_value()
    catalog = planner_catalog()
    planner = StructuredPlanner(QwenPlannerProvider(
        key,
        model=settings.planner_model,
        base_url=settings.planner_base_url,
        timeout=settings.planner_timeout_seconds,
        temperature=settings.planner_temperature,
    ), catalog)

    result = PlanValidator(catalog, max_tasks=settings.planner_max_tasks).validate(
        planner.plan(UserQuery(query="查询 syn-user-0001 的持仓，只使用必要工具"))
    )

    assert result.valid
    assert result.decision == "execute"
    assert [task.tool_name for task in result.tasks] == ["get_portfolio_positions"]


@pytest.mark.live
@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("查询 600519.SH 在 2026-08-01 至 2026-09-01 的日线", "get_market_history"),
        ("华泰科技最新研报及评级观点", "search_research_reports"),
        ("证监会现行融资融券监管规则", "search_regulatory_knowledge"),
        ("融资融券开通业务说明", "search_business_knowledge"),
    ],
)
def test_qwen37_flash_never_emits_damaged_market_or_rag_argument_keys(query, expected_tool):
    settings = Settings()
    if settings.qwen_api_key is None:
        pytest.skip("FINANCIAL_AGENT_QWEN_API_KEY is not configured")
    catalog = planner_catalog()
    provider = QwenPlannerProvider(
        settings.qwen_api_key.get_secret_value(),
        model=settings.planner_model,
        base_url=settings.planner_base_url,
        timeout=settings.planner_timeout_seconds,
        temperature=settings.planner_temperature,
    )

    plan = StructuredPlanner(provider, catalog).plan(UserQuery(query=query))
    result = PlanValidator(catalog, max_tasks=settings.planner_max_tasks).validate(plan)
    raw = provider.last_raw_response

    assert result.valid
    assert raw is not None
    if result.tasks:
        assert [task.tool_name for task in result.tasks] == [expected_tool]
        assert all(not key.startswith(":") for key in raw["tasks"][0]["arguments"])
