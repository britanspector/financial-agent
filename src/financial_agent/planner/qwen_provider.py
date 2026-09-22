"""Qwen OpenAI-compatible adapter for structured plan generation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx

from financial_agent.planner.providers import (
    PlannerProviderResponseError,
    PlannerProviderTimeoutError,
    PlannerProviderUnavailableError,
)


class QwenPlannerProvider:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3.7-flash-2026-07-15",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: float = 30.0,
        temperature: float = 0.1,
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
        # Kept only on this short-lived adapter for diagnostic comparison.  It is
        # deliberately not logged because it can contain user query/history text.
        self._last_raw_response: dict[str, Any] | None = None

    @property
    def last_raw_response(self) -> dict[str, Any] | None:
        """Exact decoded structured-output object returned by the provider."""
        return deepcopy(self._last_raw_response)

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
                "json_schema": {"name": "structured_plan", "strict": True, "schema": response_schema},
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
            raise PlannerProviderTimeoutError("Planner request timed out") from exc
        except httpx.TransportError as exc:
            raise PlannerProviderUnavailableError("Planner provider unavailable") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise PlannerProviderUnavailableError("Planner provider unavailable")
        if response.status_code >= 400:
            raise PlannerProviderResponseError("Planner request was rejected")
        try:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise PlannerProviderResponseError("Invalid planner response") from exc
        if not isinstance(result, dict):
            raise PlannerProviderResponseError("Invalid planner response")
        self._last_raw_response = deepcopy(result)
        return result
