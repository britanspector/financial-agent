import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import SecretStr

from financial_agent.config import Settings
from financial_agent.knowledge.dense import DenseIndex
from financial_agent.knowledge.ingestion import KnowledgeIngestor
from financial_agent.knowledge.models import (
    BusinessSearchInput,
    RegulatorySearchInput,
    ResearchSearchInput,
)
from financial_agent.knowledge.providers import (
    EmbeddingDescriptor,
    ProviderPermissionDeniedError,
    ProviderTimeoutError,
    RerankResult,
)
from financial_agent.knowledge.retrieval import HybridKnowledgeRetriever
from financial_agent.knowledge.runtime import build_rag_index, build_rag_tools, register_rag_tools
from financial_agent.knowledge.service import KnowledgeRetrievalService
from financial_agent.knowledge.tokenization import ChineseTokenizer
from financial_agent.user_data.auth import CallContext


CORPUS = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "manifest.json"


class HashEmbeddingProvider:
    def __init__(self, dimension=256):
        self._descriptor = EmbeddingDescriptor("offline-hash-v1", dimension)
        self._tokenizer = ChineseTokenizer()

    @property
    def descriptor(self):
        return self._descriptor

    def _embed(self, text):
        vector = np.zeros(self.descriptor.dimension, dtype=np.float32)
        for token in self._tokenizer.tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.descriptor.dimension] += 1.0
        return vector

    def embed_documents(self, texts):
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text):
        return self._embed(text)


class LexicalReranker:
    def __init__(self):
        self.calls = []
        self._tokenizer = ChineseTokenizer()

    def rerank(self, query, documents, *, top_n):
        self.calls.append((len(documents), top_n))
        query_terms = set(self._tokenizer.tokenize(query))
        scored = []
        for index, document in enumerate(documents):
            terms = self._tokenizer.tokenize(document)
            score = sum(terms.count(term) for term in query_terms) / max(len(terms), 1)
            scored.append(RerankResult(index, score))
        return sorted(scored, key=lambda item: (-item.score, item.index))[:top_n]


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, *, top_n):
        del query
        self.calls.append((len(documents), top_n))
        return [RerankResult(index, float(top_n - index)) for index in range(top_n)]


def make_retriever(reranker=None, embedding=None):
    chunks = KnowledgeIngestor().ingest(CORPUS)
    embedding = embedding or HashEmbeddingProvider()
    vectors = embedding.embed_documents([f"{chunk.title}\n{chunk.content}" for chunk in chunks])
    reranker = reranker or LexicalReranker()
    return HybridKnowledgeRetriever(chunks, DenseIndex([chunk.chunk_id for chunk in chunks], vectors), embedding, reranker), reranker


@pytest.mark.parametrize(
    ("query", "expected_call"),
    [
        ("比较工业自动化行业观点", (12, 5)),
        ("比较澄海智造和曜石半导体", (20, 8)),
        ("比较澄海智造、曜石半导体与青禾医药", (30, 10)),
    ],
)
def test_research_adaptive_top_k_has_three_tiers(query, expected_call):
    reranker = RecordingReranker()
    retriever, _ = make_retriever(reranker=reranker)

    evidence = retriever.search_research(ResearchSearchInput(query=query))

    assert reranker.calls[-1] == expected_call
    assert len(evidence) == expected_call[1]


def test_research_metadata_and_as_of_filters():
    retriever, _ = make_retriever()

    evidence = retriever.search_research(ResearchSearchInput(
        query="订单与利润率",
        companies=["澄海智造"],
        brokers=["云岫证券研究所"],
        as_of="2025-12-31",
    ))

    assert evidence
    assert all(item.metadata.company == "澄海智造" for item in evidence)
    assert all(item.metadata.broker == "云岫证券研究所" for item in evidence)
    assert all(item.metadata.publish_date.isoformat() <= "2025-12-31" for item in evidence)


def test_regulatory_and_business_filters():
    retriever, _ = make_retriever()

    regulatory = retriever.search_regulatory(RegulatorySearchInput(
        query="保证金比例",
        issuer="中证融资服务中心",
        as_of="2025-06-01",
    ))
    business = retriever.search_business(BusinessSearchInput(
        query="预警线怎么办",
        category="margin",
        as_of="2025-12-31",
    ))

    assert regulatory
    assert all(item.metadata.issuer == "中证融资服务中心" for item in regulatory)
    assert all(item.metadata.effective_date.isoformat() <= "2025-06-01" for item in regulatory)
    assert business
    assert all(item.metadata.category == "margin" for item in business)
    assert all(item.metadata.effective_date.isoformat() <= "2025-12-31" for item in business)


def test_three_tool_contracts_return_only_evidence_lists():
    retriever, _ = make_retriever()
    registry = register_rag_tools(KnowledgeRetrievalService(retriever))
    context = CallContext()
    calls = {
        "search_research_reports": {"query": "澄海智造利润率"},
        "search_regulatory_knowledge": {"query": "保证金比例"},
        "search_business_knowledge": {"query": "交易密码被锁"},
    }

    for tool, arguments in calls.items():
        result = registry.invoke(tool, arguments, context=context)
        assert result.status == "success"
        assert isinstance(result.data, list)
        assert result.data
        assert all(item.content and item.document_id and item.score >= 0 for item in result.data)


class FailingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def embed_query(self, text):
        del text
        raise self.error


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (ProviderTimeoutError("embedding", "secret-key-value"), "PROVIDER_TIMEOUT", True),
        (ProviderPermissionDeniedError("embedding", "secret-key-value"), "PROVIDER_PERMISSION_DENIED", False),
    ],
)
def test_provider_errors_are_mapped_without_secret_leakage(error, code, retryable):
    provider = FailingEmbeddingProvider(error)
    retriever, _ = make_retriever(embedding=provider)
    registry = register_rag_tools(KnowledgeRetrievalService(retriever))

    result = registry.invoke("search_business_knowledge", {"query": "密码"}, context=CallContext())

    assert result.status == "error"
    assert result.error.code == code
    assert result.error.retryable is retryable
    assert "secret-key-value" not in result.model_dump_json()


class FailingReranker:
    def rerank(self, query, documents, *, top_n):
        del query, documents, top_n
        raise ProviderTimeoutError("rerank", "secret-key-value")


def test_reranker_error_is_mapped_without_provider_details():
    retriever, _ = make_retriever(reranker=FailingReranker())
    registry = register_rag_tools(KnowledgeRetrievalService(retriever))

    result = registry.invoke("search_research_reports", {"query": "澄海智造"}, context=CallContext())

    assert result.status == "error"
    assert result.error.code == "PROVIDER_TIMEOUT"
    assert result.error.message == "Reranker provider timed out"
    assert "secret-key-value" not in result.model_dump_json()


def test_settings_never_serialize_model_key():
    settings = Settings(model_api_key=SecretStr("secret-key-value"))

    assert "secret-key-value" not in settings.model_dump_json()


def test_build_index_and_runtime_use_persisted_embeddings(tmp_path):
    provider = HashEmbeddingProvider()
    settings = Settings(
        knowledge_manifest_path=CORPUS,
        rag_embedding_index_path=tmp_path / "embeddings.npz",
    )

    count, state = build_rag_index(settings, provider)
    registry = build_rag_tools(
        settings,
        embedding_provider=provider,
        reranker_provider=LexicalReranker(),
    )
    result = registry.invoke(
        "search_business_knowledge",
        {"query": "价格提醒自动下单"},
        context=CallContext(),
    )

    assert count == 96
    assert state == "ready"
    assert result.status == "success"
