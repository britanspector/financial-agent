"""Provider boundary for structured history summaries."""

from __future__ import annotations

from typing import Any, Protocol


class SummaryProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class SummaryProviderError(RuntimeError):
    pass


class SummaryProviderTimeoutError(SummaryProviderError):
    pass


class SummaryProviderUnavailableError(SummaryProviderError):
    pass


class SummaryProviderResponseError(SummaryProviderError):
    pass
