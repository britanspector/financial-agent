import json

import pytest
from pydantic import ValidationError

from financial_agent.demo_faults import FaultSequence
from financial_agent.tools.contracts import ToolResult
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.models import CustomerContext

TOOLS = [
    "get_customer_context", "get_margin_account", "get_portfolio_positions", "get_portfolio_analytics",
]


def invoke(system, tool, arguments=None, key="test-only-full-key"):
    return system.registry.invoke(tool, arguments if arguments is not None else {"user_id": "syn-user-0001"}, context=CallContext(api_key=key))


@pytest.mark.parametrize("tool", TOOLS)
def test_each_tool_returns_typed_success_and_one_audit(make_system, tool):
    system = make_system()
    result = invoke(system, tool)
    assert result.status == "success" and result.error is None
    assert result.source == "synthetic_user_db" and result.latency >= 0
    assert len(system.repository.calls) == 1
    assert result.data.user_id == "syn-user-0001"
    event, = [json.loads(line) for line in system.audit_path.read_text().splitlines()]
    assert event["request_id"] == str(result.request_id)
    assert event["tool"] == tool and event["code"] == "SUCCESS"
    assert event["principal_id"] == "synthetic-full"
    assert event["user_id"] == "syn-user-0001"


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("key,arguments,status,code", [
    (None, {"user_id": "syn-user-0001"}, 401, "UNAUTHORIZED"),
    ("invalid-test-key", {"user_id": "syn-user-0001"}, 401, "UNAUTHORIZED"),
    ("test-only-no-scope", {"user_id": "syn-user-0001"}, 403, "FORBIDDEN"),
    ("test-only-other-user", {"user_id": "syn-user-0001"}, 403, "FORBIDDEN"),
    ("test-only-full-key", {"user_id": ""}, 422, "INVALID_ARGUMENT"),
    ("test-only-full-key", {"user_id": "syn-user-999"}, 404, "NOT_FOUND"),
])
def test_each_tool_classifies_errors(make_system, tool, key, arguments, status, code):
    system = make_system()
    result = invoke(system, tool, arguments, key)
    assert result.status == "error" and result.data is None
    assert result.error.http_status == status and result.error.code == code
    assert len(system.repository.calls) == (1 if status == 404 else 0)
    audit = json.loads(system.audit_path.read_text())
    assert audit["http_status"] == status and audit["code"] == code


@pytest.mark.parametrize("tool", TOOLS)
def test_fault_sequence_is_ordered_retryable_and_does_not_sleep(make_system, tool, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *args: pytest.fail("Fault simulation must not sleep"))
    times = iter([0.0, 0.025] * 5)
    system = make_system(fault=FaultSequence(["timeout", "429", "503", "success"]), clock=lambda: next(times))
    for code, http in [("TIMEOUT", 504), ("RATE_LIMITED", 429), ("TEMPORARY_FAILURE", 503)]:
        result = invoke(system, tool)
        assert result.error.code == code and result.error.http_status == http and result.error.retryable
        assert system.repository.calls == []
    assert invoke(system, tool).status == "success"
    assert invoke(system, tool).status == "success"
    assert len(system.repository.calls) == 2


@pytest.mark.parametrize("tool", TOOLS)
def test_scope_is_independent_per_business_tool(make_system, tool):
    key_by_tool = {
        "get_customer_context": "test-only-context-key",
        "get_margin_account": "test-only-margin-key",
        "get_portfolio_positions": "test-only-positions-key",
        "get_portfolio_analytics": "test-only-analytics-key",
    }
    assert invoke(make_system(), tool, key=key_by_tool[tool]).status == "success"
    other = next(key for name, key in key_by_tool.items() if name != tool)
    assert invoke(make_system(), tool, key=other).error.http_status == 403


@pytest.mark.parametrize("arguments", [{}, {"user_id": 123}, {"user_id": " "}, [], "bad-input"])
def test_bad_argument_shape_returns_result_not_exception(make_system, arguments):
    assert invoke(make_system(), "get_customer_context", arguments).error.code == "INVALID_ARGUMENT"


@pytest.mark.parametrize("arguments", [
    {"start_date": "2026-06-30", "end_date": "2026-05-01"},
    {"start_date": "not-a-date"}, {"limit": 0}, {"limit": 367}, {"offset": -1},
])
def test_invalid_margin_parameters(make_system, arguments):
    result = invoke(make_system(), "get_margin_account", {"user_id": "syn-user-0001", **arguments})
    assert result.error.code == "INVALID_ARGUMENT"


def test_margin_date_range_and_pagination(make_system):
    system = make_system()
    result = invoke(system, "get_margin_account", {"user_id": "syn-user-0001", "start_date": "2026-06-01", "end_date": "2026-06-30", "limit": 2})
    assert result.status == "success" and result.data.total == 29 and len(result.data.daily) == 2
    assert result.data.daily[0].trade_date == "2026-06-01"


def test_registry_schema_and_old_tools_are_gone(make_system):
    descriptions = make_system().registry.describe()
    assert [item["name"] for item in descriptions] == TOOLS
    for item in descriptions:
        assert not item["input_schema"]["additionalProperties"]
        assert set(item["input_schema"]["properties"]).isdisjoint({"api_key", "scopes", "fault"})
    result = invoke(make_system(), "unknown-secret-tool-name")
    assert result.error.code == "UNKNOWN_TOOL"


def test_customer_context_permission_fields_and_typed_result(make_system):
    result = invoke(make_system(), "get_customer_context")
    assert isinstance(result.data, CustomerContext)
    assert result.data.max_allowed_product_risk_level in {"R2", "R3", "R4", "R5"}
    assert isinstance(result.data.margin_enabled, bool)
    assert ToolResult[CustomerContext].model_validate_json(result.model_dump_json()) == result


def test_audit_is_redacted_and_requests_have_unique_ids(make_system):
    system = make_system()
    first = invoke(system, "get_customer_context")
    second = invoke(system, "get_customer_context", {"user_id": "syn-user-0001", "secret": "arbitrary-secret"})
    assert first.request_id != second.request_id
    text = system.audit_path.read_text()
    assert "test-only-full-key" not in text and "arbitrary-secret" not in text
    assert "Synthetic Client" not in text


@pytest.mark.parametrize("tool", TOOLS)
def test_audit_failure_withholds_result(make_system, tool):
    class BrokenAudit:
        def write(self, event):
            raise OSError("sensitive-path-and-test-key")
    result = invoke(make_system(audit=BrokenAudit()), tool)
    assert result.status == "error" and result.data is None
    assert result.error.code == "AUDIT_UNAVAILABLE"
    assert "sensitive-path" not in result.model_dump_json()


def test_unexpected_adapter_error_is_sanitized(make_system):
    class BrokenRepository:
        def get_customer_context(self, user_id):
            raise RuntimeError("sensitive-path-and-test-key")
    result = invoke(make_system(repo=BrokenRepository()), "get_customer_context")
    assert result.error.code == "INTERNAL_ERROR"
    assert "sensitive-path" not in result.model_dump_json()


def test_invalid_result_envelope_is_rejected():
    from uuid import uuid4
    common = dict(source="synthetic_user_db", latency=0, request_id=uuid4())
    with pytest.raises(ValidationError):
        ToolResult(status="error", data=None, error=None, **common)
    with pytest.raises(ValidationError):
        ToolResult(status="success", data=None, error=None, **common)
