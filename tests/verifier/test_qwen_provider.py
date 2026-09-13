import json

import httpx
import pytest

from financial_agent.verifier.providers import (
    VerifierProviderResponseError,
    VerifierProviderTimeoutError,
    VerifierProviderUnavailableError,
)
from financial_agent.verifier.qwen_provider import QwenVerifierProvider


def test_qwen_verifier_uses_strict_schema_temperature_zero_and_hides_key():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "decision": "PASS", "reason": "enough", "missing_evidence": [],
        })}}]})

    provider = QwenVerifierProvider(
        "test-secret", client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    schema = {"type": "object"}
    result = provider.generate([{"role": "user", "content": "verify"}], response_schema=schema)
    body = json.loads(captured["request"].content)

    assert result["decision"] == "PASS"
    assert body["temperature"] == 0.0
    assert body["enable_thinking"] is False
    assert body["response_format"]["json_schema"]["name"] == "structured_verification"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "test-secret" not in repr(provider)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_qwen_verifier_maps_transient_status_to_unavailable(status):
    provider = QwenVerifierProvider(
        "key", client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, request=request),
        )),
    )
    with pytest.raises(VerifierProviderUnavailableError):
        provider.generate([], response_schema={})


def test_qwen_verifier_maps_timeout_and_invalid_response():
    def timeout(request):
        raise httpx.ReadTimeout("late", request=request)

    provider = QwenVerifierProvider(
        "key", client=httpx.Client(transport=httpx.MockTransport(timeout)),
    )
    with pytest.raises(VerifierProviderTimeoutError):
        provider.generate([], response_schema={})

    invalid = QwenVerifierProvider(
        "key", client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, json={"choices": []}),
        )),
    )
    with pytest.raises(VerifierProviderResponseError):
        invalid.generate([], response_schema={})


def test_qwen_verifier_maps_non_transient_rejection_to_response_error():
    provider = QwenVerifierProvider(
        "key", client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, request=request),
        )),
    )
    with pytest.raises(VerifierProviderResponseError):
        provider.generate([], response_schema={})
