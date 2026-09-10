import json
import logging
from uuid import uuid4

from fastapi.testclient import TestClient

from financial_agent.config import Settings
from financial_agent.user_data.api import create_app


def make_client(db_path, tmp_path, *, scopes=None, users=None):
    key = "http-api-test-key"
    settings = Settings(
        user_db_path=db_path,
        audit_path=tmp_path / "server-audit.jsonl",
        user_api_keys=json.dumps([{
            "api_key": key,
            "principal_id": "http-test",
            "user_ids": users or ["syn-user-0001", "syn-user-999"],
            "scopes": scopes or [
                "read:customer_context", "read:margin_account",
                "read:portfolio_positions", "read:portfolio_analytics",
            ],
        }]),
    )
    return TestClient(create_app(settings)), key, settings


def test_api_returns_domain_model_and_request_id(db_path, tmp_path):
    client, key, settings = make_client(db_path, tmp_path)
    request_id = str(uuid4())
    response = client.get(
        "/v1/customers/syn-user-0001/context",
        headers={"X-API-Key": key, "X-Request-ID": request_id},
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == "syn-user-0001"
    assert "status" not in response.json()
    assert response.headers["X-Request-ID"] == request_id
    assert json.loads(settings.audit_path.read_text(encoding="utf-8").splitlines()[-1])["request_id"] == request_id


def test_api_errors_are_apieror_contract(db_path, tmp_path):
    client, key, settings = make_client(db_path, tmp_path)
    missing = client.get("/v1/customers/syn-user-0001/context")
    assert missing.status_code == 401
    assert missing.json()["code"] == "UNAUTHORIZED"
    assert "status" not in missing.json()

    not_found = client.get(
        "/v1/customers/syn-user-999/context", headers={"X-API-Key": key},
    )
    assert not_found.status_code == 404
    assert not_found.json()["code"] == "NOT_FOUND"

    invalid = client.get(
        "/v1/customers/syn-user-0001/margin-account?limit=0",
        headers={"X-API-Key": key},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "INVALID_ARGUMENT"
    assert not settings.audit_path.exists() or settings.audit_path.read_text(encoding="utf-8")


def test_api_scope_and_margin_query(db_path, tmp_path):
    client, key, _ = make_client(db_path, tmp_path, scopes=["read:customer_context"])
    forbidden = client.get(
        "/v1/customers/syn-user-0001/margin-account", headers={"X-API-Key": key},
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["code"] == "FORBIDDEN"

    full_client, full_key, _ = make_client(db_path, tmp_path)
    response = full_client.get(
        "/v1/customers/syn-user-0001/margin-account?start_date=2026-05-01&end_date=2026-05-10&limit=3&offset=1",
        headers={"X-API-Key": full_key},
    )
    assert response.status_code == 200
    assert len(response.json()["daily"]) == 3
    assert response.json()["total"] == 8


def test_api_rejects_unknown_query_parameters_without_logging_values(db_path, tmp_path, caplog):
    client, key, _ = make_client(db_path, tmp_path)
    request_id = str(uuid4())
    secret_value = "sensitive-query-value"
    paths = [
        "/v1/customers/syn-user-0001/context",
        "/v1/customers/syn-user-0001/margin-account?limit=1",
        "/v1/customers/syn-user-0001/portfolio/positions",
        "/v1/customers/syn-user-0001/portfolio/analytics",
    ]

    with caplog.at_level(logging.INFO, logger="financial_agent.user_data.api"):
        for path in paths:
            separator = "&" if "?" in path else "?"
            response = client.get(
                f"{path}{separator}unknown={secret_value}",
                headers={"X-API-Key": key, "X-Request-ID": request_id},
            )
            assert response.status_code == 422
            assert response.json()["code"] == "INVALID_ARGUMENT"
            assert response.headers["X-Request-ID"] == request_id

    assert request_id in caplog.text
    assert secret_value not in caplog.text
