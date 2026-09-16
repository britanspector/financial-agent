"""Qwen OpenAI-compatible adapter for structured history summaries."""

from __future__ import annotations

import json
from typing import Any

import httpx

from financial_agent.context.summary_providers import (
    SummaryProviderResponseError,
    SummaryProviderTimeoutError,
    SummaryProviderUnavailableError,
)


class QwenSummaryProvider:
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
        self._client = client or httpx.Client()

    def generate(self, messages: list[dict[str, str]], *, response_schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "history_summary", "strict": True, "schema": response_schema},
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
            raise SummaryProviderTimeoutError("Summary request timed out") from exc
        except httpx.TransportError as exc:
            raise SummaryProviderUnavailableError("Summary provider unavailable") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise SummaryProviderUnavailableError("Summary provider unavailable")
        if response.status_code >= 400:
            raise SummaryProviderResponseError("Summary request was rejected")
        try:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise SummaryProviderResponseError("Invalid summary response") from exc
        if not isinstance(result, dict):
            raise SummaryProviderResponseError("Invalid summary response")
        return result

    def __repr__(self) -> str:
        return f"QwenSummaryProvider(model={self._model!r}, url={self._url!r}, timeout={self._timeout!r})"
