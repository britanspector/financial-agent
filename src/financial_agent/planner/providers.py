"""Provider boundary for structured planning models."""

from __future__ import annotations

from typing import Any, Protocol


class PlannerProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class PlannerProviderError(RuntimeError):
    pass


class PlannerProviderTimeoutError(PlannerProviderError):
    pass


class PlannerProviderUnavailableError(PlannerProviderError):
    pass


class PlannerProviderResponseError(PlannerProviderError):
    pass
