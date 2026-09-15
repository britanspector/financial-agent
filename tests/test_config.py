import pytest
from pydantic import ValidationError

from financial_agent.config import Settings
from financial_agent.agent.retry import RetryPolicy
from financial_agent.agent.loop import LoopPolicy


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
    assert settings.planner_model == "qwen3.7-flash-2026-07-15"
    assert settings.planner_temperature == 0.1
    assert settings.planner_max_tasks == 12
    assert settings.verifier_model == "qwen3.7-flash"
    assert settings.verifier_temperature == 0.0
    assert settings.verifier_timeout_seconds == 30.0
    assert settings.answer_model == "qwen3.7-flash"
    assert settings.answer_temperature == 0.1
    assert settings.context_strategy == "full_history"
    assert settings.context_last_n == 6
    assert settings.context_summary_recent_n == 3
    assert settings.context_summary_budget_ratio == 0.4
    assert settings.context_summary_cache_size == 128
    assert settings.context_summary_max_facts == 24
    assert settings.context_retrieval_top_k == 4
    assert settings.context_retrieval_min_score == 0.15
    assert settings.context_retrieval_recent_reservation_ratio == 0.3
    assert settings.context_retrieval_protected_summary_reservation_ratio == 0.1
    assert settings.context_retrieval_history_reservation_ratio == 0.1
    assert settings.planner_context_budget_tokens == 4096
    assert settings.answer_context_budget_tokens == 4096
    assert settings.verifier_context_budget_tokens == 4096
    assert settings.summary_model == "qwen3.7-flash"
    assert settings.summary_temperature == 0
    assert settings.loop_max_rewrite == 2
    assert settings.loop_max_replan == 2
    assert settings.loop_max_iterations == 5
    assert settings.loop_total_tool_budget == 36
    assert settings.execution_max_retry == 2
    assert settings.execution_initial_backoff_seconds == 0.5
    assert settings.execution_backoff_multiplier == 2.0
    assert settings.execution_max_attempts == 36
    assert settings.execution_deadline_seconds == 120.0
    assert fake_key not in repr(settings)
    assert "qwen_api_key" not in settings.model_dump()
    assert fake_key not in settings.model_dump_json()


def test_flash_rejects_unsupported_embedding_dimension(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_EMBEDDING_DIMENSION", "2560")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("EXECUTION_MAX_RETRY", "-1"),
        ("EXECUTION_BACKOFF_MULTIPLIER", "0.5"),
        ("EXECUTION_MAX_ATTEMPTS", "0"),
        ("EXECUTION_DEADLINE_SECONDS", "0"),
    ],
)
def test_invalid_execution_retry_settings_rejected(monkeypatch, key, value):
    monkeypatch.setenv(f"FINANCIAL_AGENT_{key}", value)
    with pytest.raises(ValidationError):
        Settings()


def test_retry_policy_is_built_from_execution_settings(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AGENT_EXECUTION_MAX_RETRY", "3")
    monkeypatch.setenv("FINANCIAL_AGENT_EXECUTION_INITIAL_BACKOFF_SECONDS", "0.25")
    monkeypatch.setenv("FINANCIAL_AGENT_EXECUTION_BACKOFF_MULTIPLIER", "3")
    monkeypatch.setenv("FINANCIAL_AGENT_EXECUTION_MAX_ATTEMPTS", "20")
    monkeypatch.setenv("FINANCIAL_AGENT_EXECUTION_DEADLINE_SECONDS", "45")

    policy = RetryPolicy.from_settings(Settings())

    assert policy == RetryPolicy(
        max_retry=3,
        initial_backoff_seconds=0.25,
        backoff_multiplier=3,
        max_attempts=20,
        deadline_seconds=45,
    )


def test_loop_policy_is_built_from_settings(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AGENT_LOOP_MAX_REWRITE", "1")
    monkeypatch.setenv("FINANCIAL_AGENT_LOOP_MAX_REPLAN", "3")
    monkeypatch.setenv("FINANCIAL_AGENT_LOOP_MAX_ITERATIONS", "7")
    monkeypatch.setenv("FINANCIAL_AGENT_LOOP_TOTAL_TOOL_BUDGET", "19")
    assert LoopPolicy.from_settings(Settings()) == LoopPolicy(
        max_rewrite=1, max_replan=3, max_iterations=7, total_tool_budget=19,
    )


@pytest.mark.parametrize(("key", "value"), [
    ("LOOP_MAX_REWRITE", "-1"), ("LOOP_MAX_REPLAN", "-1"),
    ("LOOP_MAX_ITERATIONS", "0"), ("LOOP_TOTAL_TOOL_BUDGET", "0"),
    ("ANSWER_TIMEOUT_SECONDS", "0"), ("ANSWER_TEMPERATURE", "3"),
])
def test_invalid_agent_loop_settings_rejected(monkeypatch, key, value):
    monkeypatch.setenv(f"FINANCIAL_AGENT_{key}", value)
    with pytest.raises(ValidationError):
        Settings()


def test_context_retrieval_reservations_cannot_exceed_total_budget():
    with pytest.raises(ValidationError, match="reservation ratios"):
        Settings(
            context_retrieval_recent_reservation_ratio=0.6,
            context_retrieval_protected_summary_reservation_ratio=0.3,
            context_retrieval_history_reservation_ratio=0.2,
        )


@pytest.mark.parametrize(("key", "value"), [
    ("CONTEXT_STRATEGY", "unknown"),
    ("CONTEXT_LAST_N", "-1"),
    ("CONTEXT_SUMMARY_RECENT_N", "-1"),
    ("CONTEXT_SUMMARY_BUDGET_RATIO", "1.1"),
    ("CONTEXT_SUMMARY_CACHE_SIZE", "-1"),
    ("CONTEXT_SUMMARY_MAX_FACTS", "0"),
    ("PLANNER_CONTEXT_BUDGET_TOKENS", "-1"),
    ("ANSWER_CONTEXT_BUDGET_TOKENS", "-1"),
    ("VERIFIER_CONTEXT_BUDGET_TOKENS", "-1"),
])
def test_invalid_context_settings_rejected(monkeypatch, key, value):
    monkeypatch.setenv(f"FINANCIAL_AGENT_{key}", value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("VERIFIER_MODEL", ""),
        ("VERIFIER_TIMEOUT_SECONDS", "0"),
        ("VERIFIER_TEMPERATURE", "-0.1"),
        ("VERIFIER_TEMPERATURE", "2.1"),
    ],
)
def test_invalid_verifier_settings_rejected(monkeypatch, key, value):
    monkeypatch.setenv(f"FINANCIAL_AGENT_{key}", value)
    with pytest.raises(ValidationError):
        Settings()
