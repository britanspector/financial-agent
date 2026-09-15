import pytest

from financial_agent.config import Settings
from financial_agent.verifier.qwen_provider import QwenVerifierProvider
from financial_agent.verifier.runtime import build_verifier


class Catalog:
    def output_model(self, name):
        del name
        return None


class Provider:
    def generate(self, messages, *, response_schema):
        del messages, response_schema
        return {"decision": "PASS", "reason": "enough", "missing_evidence": []}


def test_build_verifier_accepts_injected_provider_without_key():
    verifier = build_verifier(Settings(qwen_api_key=None), Catalog(), provider=Provider())
    assert verifier is not None


def test_build_verifier_requires_key_for_default_qwen_provider():
    with pytest.raises(ValueError, match="Qwen API key is required"):
        build_verifier(Settings(qwen_api_key=None), Catalog())


def test_build_verifier_maps_settings_to_qwen_adapter(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AGENT_QWEN_API_KEY", "test-key")
    monkeypatch.setenv("FINANCIAL_AGENT_VERIFIER_MODEL", "verifier-model")
    monkeypatch.setenv("FINANCIAL_AGENT_VERIFIER_TIMEOUT_SECONDS", "12")
    verifier = build_verifier(Settings(), Catalog())

    assert isinstance(verifier._provider, QwenVerifierProvider)
    assert "verifier-model" in repr(verifier._provider)
    assert "timeout=12.0" in repr(verifier._provider)
    assert "test-key" not in repr(verifier._provider)
