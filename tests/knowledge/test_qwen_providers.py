import json

import httpx
import numpy as np
import pytest

from financial_agent.knowledge.providers import (
    ProviderPermissionDeniedError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from financial_agent.knowledge.qwen_providers import QwenEmbeddingProvider, QwenRerankerProvider


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_qwen_embedding_uses_document_and_query_modes_without_leaking_key():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        vectors = [
            {"text_index": index, "embedding": [float(index + 1), 0.0, 1.0]}
            for index, _ in enumerate(body["input"]["texts"])
        ]
        return httpx.Response(200, json={"output": {"embeddings": vectors}})

    provider = QwenEmbeddingProvider(
        "super-secret-key",
        model="qwen3.7-text-embedding",
        dimension=3,
        base_url="https://example.test/api/v1/",
        query_instruct="Retrieve finance passages.",
        client=client_for(handler),
    )

    documents = provider.embed_documents(["文档一", "文档二"])
    query = provider.embed_query("查询")

    assert documents.shape == (2, 3)
    assert query.shape == (3,)
    first = json.loads(requests[0].content)
    second = json.loads(requests[1].content)
    assert first["parameters"]["text_type"] == "document"
    assert "instruct" not in first["parameters"]
    assert second["parameters"]["text_type"] == "query"
    assert second["parameters"]["instruct"] == "Retrieve finance passages."
    assert requests[0].url.path == "/api/v1/services/embeddings/text-embedding/text-embedding"
    assert requests[0].headers["Authorization"] == "Bearer super-secret-key"
    assert "super-secret-key" not in repr(provider)


def test_qwen_flash_omits_unsupported_output_type():
    def handler(request):
        body = json.loads(request.content)
        assert "output_type" not in body["parameters"]
        return httpx.Response(200, json={"output": {"embeddings": [{"text_index": 0, "embedding": [1, 0]}]}})

    provider = QwenEmbeddingProvider(
        "secret",
        model="qwen3.7-text-embedding-flash",
        dimension=2,
        base_url="https://example.test/api/v1",
        client=client_for(handler),
    )

    assert np.array_equal(provider.embed_query("query"), np.array([1, 0], dtype=np.float32))


def test_qwen_reranker_parses_dashscope_response():
    def handler(request):
        body = json.loads(request.content)
        assert body == {
            "model": "qwen3.7-text-rerank",
            "input": {"query": "保证金", "documents": ["无关", "保证金比例"]},
            "parameters": {"top_n": 2, "instruct": "Find answers."},
        }
        return httpx.Response(200, json={"output": {"results": [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.12},
        ]}})

    provider = QwenRerankerProvider(
        "secret",
        model="qwen3.7-text-rerank",
        base_url="https://example.test/api/v1",
        instruct="Find answers.",
        client=client_for(handler),
    )

    results = provider.rerank("保证金", ["无关", "保证金比例"], top_n=2)

    assert [(item.index, item.score) for item in results] == [(1, 0.91), (0, 0.12)]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, ProviderPermissionDeniedError),
        (429, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (400, ProviderResponseError),
    ],
)
def test_qwen_http_errors_map_without_response_details(status, error):
    client = client_for(lambda request: httpx.Response(status, json={"message": "super-secret-key"}))
    provider = QwenEmbeddingProvider(
        "super-secret-key",
        model="qwen3.7-text-embedding",
        dimension=2,
        base_url="https://example.test/api/v1",
        client=client,
    )

    with pytest.raises(error) as captured:
        provider.embed_query("query")

    assert "super-secret-key" not in str(captured.value)


def test_qwen_provider_rejects_invalid_shape():
    client = client_for(lambda request: httpx.Response(200, json={"output": {"embeddings": [
        {"text_index": 0, "embedding": [1.0]},
    ]}}))
    provider = QwenEmbeddingProvider(
        "secret",
        model="qwen3.7-text-embedding",
        dimension=2,
        base_url="https://example.test/api/v1",
        client=client,
    )

    with pytest.raises(ProviderResponseError):
        provider.embed_query("query")
