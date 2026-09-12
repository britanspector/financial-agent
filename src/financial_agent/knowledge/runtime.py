"""Composition helpers for the three independent RAG tools."""

from __future__ import annotations

from financial_agent.config import Settings
from financial_agent.knowledge.index import EmbeddingIndexStore
from financial_agent.knowledge.ingestion import KnowledgeIngestor
from financial_agent.knowledge.models import (
    BusinessSearchInput,
    EvidenceList,
    RegulatorySearchInput,
    ResearchSearchInput,
)
from financial_agent.knowledge.providers import EmbeddingProvider, RerankerProvider
from financial_agent.knowledge.qwen_providers import QwenEmbeddingProvider, QwenRerankerProvider
from financial_agent.knowledge.retrieval import HybridKnowledgeRetriever
from financial_agent.knowledge.service import KnowledgeRetrievalService
from financial_agent.tools.registry import ToolRegistry, ToolSpec


def register_rag_tools(service: KnowledgeRetrievalService) -> ToolRegistry:
    registry = ToolRegistry(service)
    registry.register(ToolSpec(
        "search_research_reports",
        "Search research-report evidence with company, broker, and publication-time filters; reports are not real-time news",
        ResearchSearchInput,
        EvidenceList,
        None,
        "search_research_reports",
    ))
    registry.register(ToolSpec(
        "search_regulatory_knowledge",
        "Search regulatory knowledge for laws, regulator announcements, supervision, and suitability rules with issuer and effective-time filters",
        RegulatorySearchInput,
        EvidenceList,
        None,
        "search_regulatory_knowledge",
    ))
    registry.register(ToolSpec(
        "search_business_knowledge",
        "Search business FAQ knowledge for business processes and product explanations with category and effective-time filters",
        BusinessSearchInput,
        EvidenceList,
        None,
        "search_business_knowledge",
    ))
    return registry


def build_rag_tools(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
) -> ToolRegistry:
    """Build the three RAG tools from persisted embeddings and configured providers."""
    embedding_provider = embedding_provider or _build_embedding_provider(settings)
    reranker_provider = reranker_provider or _build_reranker_provider(settings)
    chunks = KnowledgeIngestor(max_chars=settings.rag_chunk_max_chars).ingest(settings.knowledge_manifest_path)
    dense_index = EmbeddingIndexStore(settings.rag_embedding_index_path).load(
        chunks, embedding_provider.descriptor,
    )
    retriever = HybridKnowledgeRetriever(
        chunks,
        dense_index,
        embedding_provider,
        reranker_provider,
        rrf_k=settings.rag_rrf_k,
    )
    return register_rag_tools(KnowledgeRetrievalService(retriever))


def build_rag_index(
    settings: Settings, embedding_provider: EmbeddingProvider | None = None
) -> tuple[int, str]:
    """Build and persist all corpus embeddings."""
    embedding_provider = embedding_provider or _build_embedding_provider(settings)
    chunks = KnowledgeIngestor(max_chars=settings.rag_chunk_max_chars).ingest(settings.knowledge_manifest_path)
    store = EmbeddingIndexStore(settings.rag_embedding_index_path)
    store.build(chunks, embedding_provider, batch_size=settings.qwen_embedding_batch_size)
    return len(chunks), store.state(chunks, embedding_provider.descriptor)


def _build_embedding_provider(settings: Settings) -> QwenEmbeddingProvider:
    if settings.qwen_api_key is None:
        raise ValueError("Qwen API key is required")
    return QwenEmbeddingProvider(
        settings.qwen_api_key.get_secret_value(),
        model=settings.qwen_embedding_model,
        dimension=settings.qwen_embedding_dimension,
        base_url=settings.qwen_embedding_base_url,
        timeout=settings.qwen_timeout_seconds,
        query_instruct=settings.qwen_embedding_query_instruct,
    )


def _build_reranker_provider(settings: Settings) -> QwenRerankerProvider:
    if settings.qwen_api_key is None:
        raise ValueError("Qwen API key is required")
    return QwenRerankerProvider(
        settings.qwen_api_key.get_secret_value(),
        model=settings.qwen_reranker_model,
        base_url=settings.qwen_reranker_base_url,
        timeout=settings.qwen_timeout_seconds,
        instruct=settings.qwen_reranker_instruct,
    )
