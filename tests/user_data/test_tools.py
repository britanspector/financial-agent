import json

import pytest
from pydantic import ValidationError

from financial_agent.demo_faults import FaultSequence
from financial_agent.tools.contracts import ToolResult
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.models import Profile

TOOLS = ["get_user_profile", "get_user_portfolio", "get_user_transactions"]


def invoke(system, tool, arguments=None, key="test-only-full-key"):
    return system.registry.invoke(tool, arguments if arguments is not None else {"user_id": "syn-user-001"},
                                  context=CallContext(api_key=key))


@pytest.mark.parametrize("tool", TOOLS)
def test_each_tool_returns_typed_success_and_one_audit(make_system, tool):
    system = make_system()
    result = invoke(system, tool)
    assert result.status == "success" and result.error is None
    assert result.source == "synthetic_user_db" and result.latency >= 0
    assert len(system.repository.calls) == 1
    assert result.data.user_id == "syn-user-001"
    event, = [json.loads(line) for line in system.audit_path.read_text().splitlines()]
    assert event["request_id"] == str(result.request_id)
    assert event["tool"] == tool and event["code"] == "SUCCESS"
    assert event["principal_id"] == "synthetic-full"
    assert event["user_id"] == "syn-user-001"
    assert event["latency"] == result.latency


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("key,arguments,status,code", [
    (None, {"user_id": "syn-user-001"}, 401, "UNAUTHORIZED"),
    ("invalid-test-key", {"user_id": "syn-user-001"}, 401, "UNAUTHORIZED"),
    ("test-only-no-scope", {"user_id": "syn-user-001"}, 403, "FORBIDDEN"),
    ("test-only-other-user", {"user_id": "syn-user-001"}, 403, "FORBIDDEN"),
    ("test-only-full-key", {"user_id": ""}, 422, "INVALID_ARGUMENT"),
    ("test-only-full-key", {"user_id": "syn-user-999"}, 404, "NOT_FOUND"),
])
def test_each_tool_classifies_errors(make_system, tool, key, arguments, status, code):
    system = make_system()
    result = invoke(system, tool, arguments, key)
    assert result.status == "error" and result.data is None
    assert result.error.http_status == status and result.error.code == code
    assert not result.error.retryable
    assert len(system.repository.calls) == (1 if status == 404 else 0)
    audit = json.loads(system.audit_path.read_text())
    assert audit["http_status"] == status and audit["code"] == code
    if status == 401:
        assert audit["principal_id"] is None


@pytest.mark.parametrize("tool", TOOLS)
def test_fault_sequence_is_ordered_retryable_and_does_not_sleep(make_system, tool, monkeypatch):
    def reject_sleep(*args):
        pytest.fail("Fault simulation must not sleep")
    monkeypatch.setattr("time.sleep", reject_sleep)
    times = iter([0.0, 0.025] * 5)
    system = make_system(fault=FaultSequence(["timeout", "429", "503", "success"]), clock=lambda: next(times))
    for code, http in [("TIMEOUT", 504), ("RATE_LIMITED", 429), ("TEMPORARY_FAILURE", 503)]:
        result = invoke(system, tool)
        assert result.status == "error" and result.data is None
        assert result.error.code == code and result.error.http_status == http
        assert result.error.retryable and result.latency == 25
        assert system.repository.calls == []
    assert invoke(system, tool).status == "success"
    assert invoke(system, tool).status == "success"
    assert len(system.repository.calls) == 2
    assert len(system.audit_path.read_text().splitlines()) == 5


@pytest.mark.parametrize("arguments,key,expected", [
    ({"user_id": ""}, "invalid-test-key", 401),
    ({"user_id": ""}, "test-only-no-scope", 403),
    ({"user_id": ""}, "test-only-full-key", 422),
    ({"user_id": "syn-user-888"}, "test-only-full-key", 403),
])
def test_auth_and_validation_precede_fault_and_repository(make_system, arguments, key, expected):
    system = make_system(fault=FaultSequence(["503"]))
    result = invoke(system, "get_user_profile", arguments, key)
    assert result.error.http_status == expected
    assert system.repository.calls == []
    assert invoke(system, "get_user_profile").error.code == "TEMPORARY_FAILURE"


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("extra", [{"api_key": "test-only-full-key"}, {"scopes": ["read:profile"]},
                                  {"fault": "success"}, {"user_ids": ["syn-user-001"]}])
def test_security_controls_are_not_tool_arguments(make_system, tool, extra):
    system = make_system()
    result = invoke(system, tool, {"user_id": "syn-user-001", **extra})
    assert result.error.http_status == 422
    assert system.repository.calls == []


@pytest.mark.parametrize("arguments", [
    {}, {"user_id": 123}, {"user_id": " "}, [], "bad-input",
])
def test_bad_argument_shape_returns_result_not_exception(make_system, arguments):
    result = invoke(make_system(), "get_user_profile", arguments)
    assert result.error.code == "INVALID_ARGUMENT"


@pytest.mark.parametrize("arguments", [
    {"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "20"}, {"offset": -1},
    {"start_time": "2026-01-01T00:00:00"}, {"end_time": "not-a-time"},
    {"start_time": "2026-01-02T00:00:00Z", "end_time": "2026-01-01T00:00:00Z"},
    {"start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:00:00Z"},
])
def test_invalid_transaction_parameters(make_system, arguments):
    system = make_system()
    result = invoke(system, "get_user_transactions", {"user_id": "syn-user-001", **arguments})
    assert result.error.http_status == 422 and system.repository.calls == []


@pytest.mark.parametrize("tool,user_id", [
    ("get_user_portfolio", "syn-user-002"), ("get_user_portfolio", "syn-user-003"),
    ("get_user_transactions", "syn-user-002"), ("get_user_transactions", "syn-user-004"),
])
def test_empty_collections_preserve_data(make_system, tool, user_id):
    result = invoke(make_system(), tool, {"user_id": user_id})
    assert result.status == "empty" and result.data is not None and result.error is None
    assert result.data.currency == "CNY"


def test_scopes_are_enforced_per_tool(make_system):
    system = make_system()
    assert invoke(system, "get_user_profile", key="test-only-profile-key").status == "success"
    assert invoke(system, "get_user_profile", key="test-only-portfolio-key").error.http_status == 403
    for tool in TOOLS[1:]:
        assert invoke(system, tool, key="test-only-profile-key").error.http_status == 403
        assert invoke(system, tool, key="test-only-portfolio-key").status == "success"


def test_registry_schema_unknown_tool_and_null_profile(make_system):
    system = make_system()
    descriptions = system.registry.describe()
    assert [item["name"] for item in descriptions] == TOOLS
    for item in descriptions:
        schema = item["input_schema"]
        assert not schema["additionalProperties"]
        assert set(schema["properties"]).isdisjoint({"api_key", "scopes", "fault"})
    result = invoke(system, "unknown-secret-tool-name")
    assert result.error.code == "UNKNOWN_TOOL"
    assert "unknown-secret-tool-name" not in system.audit_path.read_text()
    result = invoke(system, "get_user_profile", {"user_id": "syn-user-005"})
    assert result.status == "success"
    assert result.data.risk_level is None
    assert ToolResult[Profile].model_validate_json(result.model_dump_json()) == result


def test_audit_is_redacted_and_requests_have_unique_ids(make_system):
    system = make_system()
    first = invoke(system, "get_user_profile")
    second = invoke(system, "get_user_profile", {"user_id": "syn-user-001", "secret": "arbitrary-secret"})
    assert first.request_id != second.request_id
    text = system.audit_path.read_text()
    assert "test-only-full-key" not in text and "arbitrary-secret" not in text
    assert "Synthetic Alpha" not in text and "name_alias" not in text
    assert "arbitrary-secret" not in second.model_dump_json()


@pytest.mark.parametrize("tool", TOOLS)
def test_audit_failure_withholds_result(make_system, tool):
    class BrokenAudit:
        def write(self, event):
            raise OSError("sensitive-path-and-test-key")
    result = invoke(make_system(audit=BrokenAudit()), tool)
    assert result.status == "error" and result.data is None
    assert result.error.code == "AUDIT_UNAVAILABLE" and not result.error.retryable
    assert "sensitive-path" not in result.model_dump_json()


def test_unexpected_adapter_error_is_sanitized(make_system):
    class BrokenRepository:
        def get_profile(self, user_id):
            raise RuntimeError("sensitive-path-and-test-key")
    result = invoke(make_system(repo=BrokenRepository()), "get_user_profile")
    assert result.error.code == "INTERNAL_ERROR" and not result.error.retryable
    assert "sensitive-path" not in result.model_dump_json()


def test_invalid_result_envelope_is_rejected():
    from uuid import uuid4
    common = dict(source="synthetic_user_db", latency=0, request_id=uuid4())
    with pytest.raises(ValidationError):
        ToolResult(status="error", data=None, error=None, **common)
    with pytest.raises(ValidationError):
        ToolResult(status="success", data=None, error=None, **common)
    with pytest.raises(ValueError):
        FaultSequence(["unknown"])
