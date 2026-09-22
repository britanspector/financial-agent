"""Qwen OpenAI-compatible adapter for strict verification output."""

from __future__ import annotations

import json
from typing import Any

import httpx

from financial_agent.verifier.providers import (
    VerifierProviderResponseError,
    VerifierProviderTimeoutError,
    VerifierProviderUnavailableError,
)


class QwenVerifierProvider:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3.7-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: float = 30.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Qwen API key is required")
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout
        self._temperature = temperature
        self._client = client or httpx.Client(trust_env=False)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_verification",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        try:
            response = self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise VerifierProviderTimeoutError("Verifier request timed out") from exc
        except httpx.TransportError as exc:
            raise VerifierProviderUnavailableError("Verifier provider unavailable") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise VerifierProviderUnavailableError("Verifier provider unavailable")
        if response.status_code >= 400:
            raise VerifierProviderResponseError("Verifier request was rejected")
        try:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise VerifierProviderResponseError("Invalid verifier response") from exc
        if not isinstance(result, dict):
            raise VerifierProviderResponseError("Invalid verifier response")
        return result

    def __repr__(self) -> str:
        return (
            f"QwenVerifierProvider(model={self._model!r}, url={self._url!r}, "
            f"timeout={self._timeout!r}, temperature={self._temperature!r})"
        )
