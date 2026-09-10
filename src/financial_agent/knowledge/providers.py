"""Provider boundaries for remote embedding and reranking services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EmbeddingDescriptor:
    model: str
    dimension: int


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class ProviderError(RuntimeError):
    def __init__(self, operation: str, message: str = "Provider request failed") -> None:
        super().__init__(message)
        self.operation = operation


class ProviderTimeoutError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderPermissionDeniedError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class EmbeddingProvider(Protocol):
    @property
    def descriptor(self) -> EmbeddingDescriptor: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class RerankerProvider(Protocol):
    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]: ...
