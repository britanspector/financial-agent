import httpx
import pytest

from financial_agent.user_data.api import create_app
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.http_client import HttpUserDataClient
from financial_agent.user_data.runtime import register_user_tools


def http_registry(db_path, tmp_path):
    import json
    from financial_agent.config import Settings

    key = "http-client-test-key"
    settings = Settings(
        user_db_path=db_path,
        audit_path=tmp_path / "client-server-audit.jsonl",
        user_api_keys=json.dumps([{
            "api_key": key, "principal_id": "client-test",
            "user_ids": ["syn-user-0001", "syn-user-999"],
            "scopes": ["read:customer_context", "read:margin_account", "read:portfolio_positions", "read:portfolio_analytics"],
        }]),
    )
    app = create_app(settings)
    client = HttpUserDataClient(
        "http://testserver", api_key=key, transport=httpx.ASGITransport(app=app), timeout=2,
    )
    return register_user_tools(client), key


def test_http_client_restores_toolresult_for_all_tools(db_path, tmp_path):
    registry, key = http_registry(db_path, tmp_path)
    context = CallContext(api_key=key)
    for name in ["get_customer_context", "get_margin_account", "get_portfolio_positions", "get_portfolio_analytics"]:
        result = registry.invoke(name, {"user_id": "syn-user-0001"}, context=context)
        assert result.status == "success"
        assert result.data is not None
        assert "status" not in result.data.model_dump()


def test_http_client_maps_server_errors(db_path, tmp_path):
    registry, _ = http_registry(db_path, tmp_path)
    forbidden = registry.invoke(
        "get_customer_context", {"user_id": "syn-user-0001"}, context=CallContext(api_key="wrong"),
    )
    assert forbidden.error.code == "UNAUTHORIZED"
    assert forbidden.error.http_status == 401


@pytest.mark.parametrize("exception", [
    httpx.ConnectTimeout("connect timed out"),
    httpx.ReadTimeout("read timed out"),
    httpx.TimeoutException("request timed out"),
])
def test_http_client_maps_timeouts_to_504(exception):
    client = HttpUserDataClient(
        "http://testserver", api_key="key", timeout=0.001,
        transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(exception)),
    )
    registry = register_user_tools(client)
    result = registry.invoke(
        "get_customer_context", {"user_id": "syn-user-0001"}, context=CallContext(api_key="key"),
    )
    assert result.error.code == "TIMEOUT"
    assert result.error.http_status == 504
    assert result.error.retryable is True


def test_http_client_maps_connection_refused_to_503():
    client = HttpUserDataClient(
        "http://testserver", api_key="key",
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
        ),
    )
    registry = register_user_tools(client)
    result = registry.invoke(
        "get_customer_context", {"user_id": "syn-user-0001"}, context=CallContext(api_key="key"),
    )
    assert result.error.code == "DATA_UNAVAILABLE"
    assert result.error.http_status == 503
    assert result.error.retryable is True
    assert "connection refused" not in result.error.message
