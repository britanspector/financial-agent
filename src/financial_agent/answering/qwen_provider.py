"""Qwen OpenAI-compatible adapter for structured answer output."""

import json
from copy import deepcopy
from typing import Any
import httpx

from financial_agent.answering.providers import (
    AnswerProviderResponseError, AnswerProviderTimeoutError, AnswerProviderUnavailableError,
)


class QwenAnswerProvider:
    def __init__(self, api_key: str, *, model: str = "qwen3.7-flash",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 timeout: float = 30.0, temperature: float = 0.1,
                 client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("Qwen API key is required")
        self._api_key, self._model = api_key, model
        self._url, self._timeout, self._temperature = f"{base_url.rstrip('/')}/chat/completions", timeout, temperature
        self._client = client or httpx.Client(trust_env=False)
        self._last_raw_response: dict[str, Any] | None = None

    @property
    def last_raw_response(self) -> dict[str, Any] | None:
        """Return a defensive copy for local evaluation diagnostics."""
        return deepcopy(self._last_raw_response)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[dict[str, str]], *, response_schema: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self._model, "messages": messages, "temperature": self._temperature,
                   "enable_thinking": False, "response_format": {"type": "json_schema", "json_schema": {
                       "name": "structured_answer", "strict": True, "schema": response_schema}}}
        try:
            response = self._client.post(self._url, headers={"Authorization": f"Bearer {self._api_key}",
                                         "Content-Type": "application/json"}, json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise AnswerProviderTimeoutError("Answer request timed out") from exc
        except httpx.TransportError as exc:
            raise AnswerProviderUnavailableError("Answer provider unavailable") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise AnswerProviderUnavailableError("Answer provider unavailable")
        if response.status_code >= 400:
            raise AnswerProviderResponseError("Answer request was rejected")
        try:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise AnswerProviderResponseError("Invalid answer response") from exc
        if not isinstance(result, dict):
            raise AnswerProviderResponseError("Invalid answer response")
        self._last_raw_response = deepcopy(result)
        return result
