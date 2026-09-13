"""Provider boundary and failures for structured verification models."""

from __future__ import annotations

from typing import Any, Protocol


class VerifierProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class VerifierProviderError(RuntimeError):
    pass


class VerifierProviderTimeoutError(VerifierProviderError):
    pass


class VerifierProviderUnavailableError(VerifierProviderError):
    pass


class VerifierProviderResponseError(VerifierProviderError):
    pass
