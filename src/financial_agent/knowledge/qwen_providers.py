"""HTTP adapters for Qwen 3.7 embedding and text reranking."""

from __future__ import annotations

import math
from typing import Any

import httpx
import numpy as np

from financial_agent.knowledge.providers import (
    EmbeddingDescriptor,
    ProviderPermissionDeniedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RerankResult,
)


class QwenEmbeddingProvider:
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        dimension: int,
        base_url: str,
        timeout: float = 30.0,
        query_instruct: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Qwen API key is required")
        self._api_key = api_key
        self._descriptor = EmbeddingDescriptor(model=model, dimension=dimension)
        self._url = _endpoint(base_url, "services/embeddings/text-embedding/text-embedding")
        self._timeout = timeout
        self._query_instruct = query_instruct
        self._client = client or httpx.Client(trust_env=False)

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._descriptor

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: list[str], *, text_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.descriptor.dimension), dtype=np.float32)
        parameters: dict[str, Any] = {
            "dimension": self.descriptor.dimension,
            "text_type": text_type,
        }
        if self.descriptor.model != "qwen3.7-text-embedding-flash":
            parameters["output_type"] = "dense"
        if text_type == "query" and self._query_instruct:
            parameters["instruct"] = self._query_instruct
        payload = {
            "model": self.descriptor.model,
            "input": {"texts": texts},
            "parameters": parameters,
        }
        response = _post(self._client, self._url, self._api_key, payload, self._timeout, "embedding")
        try:
            body = response.json()
            records = body["output"]["embeddings"]
            ordered = sorted(records, key=lambda item: item.get("text_index", item.get("index")))
            indices = [item.get("text_index", item.get("index")) for item in ordered]
            vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderResponseError("embedding", "Invalid embedding response") from exc
        if (
            indices != list(range(len(texts)))
            or vectors.shape != (len(texts), self.descriptor.dimension)
            or not np.isfinite(vectors).all()
        ):
            raise ProviderResponseError("embedding", "Invalid embedding response shape")
        return vectors


class QwenRerankerProvider:
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str,
        timeout: float = 30.0,
        instruct: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Qwen API key is required")
        self._api_key = api_key
        self._model = model
        self._url = _endpoint(base_url, "services/rerank/text-rerank/text-rerank")
        self._timeout = timeout
        self._instruct = instruct
        self._client = client or httpx.Client(trust_env=False)

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        if not documents or top_n <= 0:
            return []
        parameters: dict[str, Any] = {"top_n": min(top_n, len(documents))}
        if self._instruct:
            parameters["instruct"] = self._instruct
        payload = {
            "model": self._model,
            "input": {"query": query, "documents": documents},
            "parameters": parameters,
        }
        response = _post(self._client, self._url, self._api_key, payload, self._timeout, "rerank")
        try:
            body = response.json()
            records = body["output"]["results"]
            results = [
                RerankResult(index=int(item["index"]), score=float(item["relevance_score"]))
                for item in records
            ]
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderResponseError("rerank", "Invalid reranker response") from exc
        indices = [result.index for result in results]
        if (
            len(results) > min(top_n, len(documents))
            or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(documents) for index in indices)
            or any(not math.isfinite(result.score) for result in results)
        ):
            raise ProviderResponseError("rerank", "Invalid reranker result count")
        return results


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path}"


def _post(
    client: httpx.Client,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    operation: str,
) -> httpx.Response:
    try:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(operation) from exc
    except httpx.TransportError as exc:
        raise ProviderUnavailableError(operation) from exc
    if response.status_code in {401, 403}:
        raise ProviderPermissionDeniedError(operation)
    if response.status_code == 429 or response.status_code >= 500:
        raise ProviderUnavailableError(operation)
    if response.status_code >= 400:
        raise ProviderResponseError(operation)
    return response
