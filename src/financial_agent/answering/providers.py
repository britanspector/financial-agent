"""Provider boundary for structured draft generation."""

from typing import Any, Protocol


class AnswerProvider(Protocol):
    def generate(self, messages: list[dict[str, str]], *, response_schema: dict[str, Any]) -> dict[str, Any]: ...


class AnswerProviderError(RuntimeError): pass
class AnswerProviderTimeoutError(AnswerProviderError): pass
class AnswerProviderUnavailableError(AnswerProviderError): pass
class AnswerProviderResponseError(AnswerProviderError): pass
