from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from pydantic import ValidationError

from financial_agent.planner.models import PlannedTask, StructuredPlan
from financial_agent.planner.prompt import build_planner_messages
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, UserQuery

from .conftest import planner_catalog


class FakePlannerProvider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return self.output


def test_prompt_contains_query_complete_history_tools_and_rules():
    catalog = planner_catalog()
    request = UserQuery(
        query="查持仓",
        history=[Message(role="user", content="用户是 syn-user-0001"), Message(role="assistant", content="收到")],
    )
    messages = build_planner_messages(request, catalog.describe(), current_date=date(2026, 9, 11))
    payload = json.loads(messages[1]["content"])

    assert payload["query"] == request.query
    assert payload["history"] == [item.model_dump(mode="json") for item in request.history]
    assert len(payload["tools"]) == 9
    assert payload["current_date"] == "2026-09-11"
    assert all("description" in item and "input_schema" in item for item in payload["tools"])
    assert "execution order only" in messages[0]["content"]
    assert "Synthetic company and institution names are valid as written" in messages[0]["content"]
    assert "make B depend on A" in messages[0]["content"]
    assert "Map 'as of/until a date' to as_of" in messages[0]["content"]
    assert "search_regulatory_knowledge" in messages[0]["content"]
    assert "FINAL CHECKLIST BEFORE OUTPUT" in messages[0]["content"]
    assert "one search task with every supplied company/broker" in messages[0]["content"]


def test_fake_provider_produces_strict_structured_plan():
    provider = FakePlannerProvider({"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "get_portfolio_positions",
        "arguments": {"user_id": "syn-user-0001"}, "dependencies": [],
    }]})
    planner = StructuredPlanner(provider, planner_catalog())

    plan = planner.plan(UserQuery(query="查持仓"))

    assert plan.tasks[0].task_id == "t1"
    assert len(provider.calls[0][1]["oneOf"]) == 3


def test_structured_plan_rejects_extra_fields():
    with pytest.raises(ValidationError):
        StructuredPlan.model_validate({"decision": "no_tool", "tasks": [], "answer": "not allowed"})


def test_qwen_adapter_sends_low_temperature_json_schema_without_leaking_key():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"decision":"no_tool","tasks":[]}'}}]})

    provider = QwenPlannerProvider(
        "test-secret", client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.generate([{"role": "user", "content": "JSON plan"}], response_schema={"type": "object"})

    body = json.loads(captured["request"].content)
    assert result == {"decision": "no_tool", "tasks": []}
    assert body["model"] == "qwen3.7-flash-2026-07-15"
    assert body["temperature"] == 0.1
    assert body["enable_thinking"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert captured["request"].headers["authorization"] == "Bearer test-secret"
    assert "test-secret" not in repr(provider)


def test_provider_preserves_raw_argument_keys_before_pydantic_validation():
    def handler(request):
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "decision": "execute", "tasks": [{
                "task_id": "t1", "tool_name": "get_market_snapshot",
                "arguments": {":symbol": "600519.SH"}, "dependencies": [],
            }],
        })}}]})

    provider = QwenPlannerProvider("test-secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    raw = provider.generate([], response_schema={"type": "object"})

    assert raw["tasks"][0]["arguments"] == {":symbol": "600519.SH"}
    assert provider.last_raw_response == raw


def test_provider_schema_binds_each_tool_to_its_public_argument_schema():
    provider = FakePlannerProvider({"decision": "no_tool", "tasks": []})
    StructuredPlanner(provider, planner_catalog()).plan(UserQuery(query="你好"))

    schema = provider.calls[0][1]
    market_branch = next(branch for branch in schema["oneOf"][0]["properties"]["tasks"]["items"]["oneOf"]
                         if branch["properties"]["tool_name"] == {"const": "get_market_history"})
    assert set(market_branch["properties"]["arguments"]["properties"]) == {
        "symbol", "start_date", "end_date",
    }
    assert market_branch["properties"]["arguments"]["additionalProperties"] is False
    research_branch = next(branch for branch in schema["oneOf"][0]["properties"]["tasks"]["items"]["oneOf"]
                           if branch["properties"]["tool_name"] == {"const": "search_research_reports"})
    assert set(research_branch["properties"]["arguments"]["required"]) == {
        "query", "companies", "brokers", "as_of",
    }


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_market_history", {"symbol": "600519.SH", "start_date": "2026-08-01", "end_date": "2026-09-01"}),
        ("search_research_reports", {"query": "华泰科技", "companies": ["华泰科技"]}),
        ("search_regulatory_knowledge", {"query": "融资融券监管规则", "issuer": "证监会"}),
        ("search_business_knowledge", {"query": "融资融券开通说明", "category": "margin"}),
    ],
)
def test_tool_aware_schema_and_fixture_plan_keep_public_argument_keys(tool_name, arguments):
    provider = FakePlannerProvider({"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": tool_name, "arguments": arguments, "dependencies": [],
    }]})
    planner = StructuredPlanner(provider, planner_catalog())

    plan = planner.plan(UserQuery(query="fixture"))
    result = PlanValidator(planner_catalog()).validate(plan)
    branches = provider.calls[0][1]["oneOf"][0]["properties"]["tasks"]["items"]["oneOf"]
    branch = next(item for item in branches if item["properties"]["tool_name"] == {"const": tool_name})

    assert set(branch["properties"]["arguments"]["properties"]) >= set(arguments)
    assert plan.tasks[0].arguments == arguments
    assert result.valid is True
