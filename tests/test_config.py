import pytest
from pydantic import ValidationError

from financial_agent.config import Settings


def test_defaults_need_no_api_key():
    settings = Settings()
    assert settings.data_mode == "synthetic"
    assert settings.log_level == "INFO"
    assert settings.model_api_key is None


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FINANCIAL_AGENT_LOG_LEVEL=DEBUG\n", encoding="utf-8")
    assert Settings().log_level == "DEBUG"
    monkeypatch.setenv("FINANCIAL_AGENT_LOG_LEVEL", "WARNING")
    assert Settings().log_level == "WARNING"


@pytest.mark.parametrize("key,value", [("DATA_MODE", "production"), ("LOG_LEVEL", "INVALID")])
def test_invalid_settings_rejected(monkeypatch, key, value):
    monkeypatch.setenv(f"FINANCIAL_AGENT_{key}", value)
    with pytest.raises(ValidationError):
        Settings()


def test_secret_is_hidden_from_repr_and_serialization(monkeypatch):
    fake_key = "synthetic-test-secret"
    monkeypatch.setenv("FINANCIAL_AGENT_MODEL_API_KEY", fake_key)
    settings = Settings()
    assert settings.model_api_key.get_secret_value() == fake_key
    assert fake_key not in repr(settings)
    assert "model_api_key" not in settings.model_dump()
    assert fake_key not in settings.model_dump_json()


def test_qwen_secret_and_model_defaults(monkeypatch):
    fake_key = "synthetic-qwen-secret"
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_API_KEY", fake_key)
    settings = Settings()

    assert settings.qwen_api_key.get_secret_value() == fake_key
    assert settings.qwen_embedding_model == "qwen3.7-text-embedding"
    assert settings.qwen_embedding_dimension == 1024
    assert settings.qwen_reranker_model == "qwen3.7-text-rerank"
    assert settings.planner_model == "qwen3.7-flash"
    assert settings.planner_temperature == 0.1
    assert settings.planner_max_tasks == 12
    assert fake_key not in repr(settings)
    assert "qwen_api_key" not in settings.model_dump()
    assert fake_key not in settings.model_dump_json()


def test_flash_rejects_unsupported_embedding_dimension(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_EMBEDDING_DIMENSION", "2560")

    with pytest.raises(ValidationError):
        Settings()
