import json

import pytest
from pydantic import SecretStr

from financial_agent.config import Settings
from financial_agent.user_data.auth import CallContext, CredentialStore
from financial_agent.user_data.runtime import build_user_tools


def test_environment_composition_and_secret_exclusion(monkeypatch, db_path, tmp_path):
    secret = "synthetic-runtime-test-key"
    records = [{"api_key": secret, "principal_id": "synthetic-runtime", "user_ids": ["syn-user-001"],
                "scopes": ["read:profile"]}]
    monkeypatch.setenv("FINANCIAL_AGENT_USER_API_KEYS", json.dumps(records))
    monkeypatch.setenv("FINANCIAL_AGENT_CALLER_API_KEY", secret)
    settings = Settings(user_db_path=db_path, audit_path=tmp_path / "runtime.jsonl")
    assert secret not in settings.model_dump_json() and secret not in repr(settings)
    context = CallContext(api_key=settings.caller_api_key)
    assert secret not in repr(context) and secret not in context.model_dump_json()
    registry = build_user_tools(settings)
    result = registry.invoke("get_user_profile", {"user_id": "syn-user-001"}, context=context)
    assert result.status == "success" and secret not in result.model_dump_json()


@pytest.mark.parametrize("value", ["broken-test-secret", "{}", "[null]", '[{"api_key":"test-secret"}]',
    '[{"api_key":"","principal_id":"test","user_ids":[],"scopes":[]}]',
    '[{"api_key":"test-secret","principal_id":"test","user_ids":[],"scopes":["admin"]}]',
])
def test_invalid_credential_configuration_is_safe(value):
    with pytest.raises(ValueError, match="Invalid user API-key configuration") as exc:
        CredentialStore.from_json(SecretStr(value))
    assert "test-secret" not in str(exc.value)


def test_duplicate_keys_are_rejected():
    record = {"api_key": "synthetic-duplicate-secret", "principal_id": "test", "user_ids": [], "scopes": []}
    with pytest.raises(ValueError, match="Invalid user API-key configuration"):
        CredentialStore.from_json(SecretStr(json.dumps([record, record])))


def test_unconfigured_runtime_denies_access(tmp_path):
    registry = build_user_tools(Settings(audit_path=tmp_path / "audit.jsonl"))
    result = registry.invoke("get_user_profile", {"user_id": "syn-user-001"}, context=CallContext())
    assert result.error.http_status == 401


def test_missing_database_is_safe_and_not_created(tmp_path):
    key = "synthetic-missing-db-key"
    settings = Settings(user_db_path=tmp_path / "absent.db", audit_path=tmp_path / "audit.jsonl",
                        user_api_keys=json.dumps([{"api_key": key, "principal_id": "test",
                            "user_ids": ["syn-user-001"], "scopes": ["read:profile"]}]))
    result = build_user_tools(settings).invoke("get_user_profile", {"user_id": "syn-user-001"}, context=CallContext(api_key=key))
    assert result.error.code == "DATA_UNAVAILABLE" and result.error.retryable
    assert "absent.db" not in result.model_dump_json()
    assert not settings.user_db_path.exists()
