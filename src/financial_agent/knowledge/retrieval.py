"""Filtering and BM25+dense+RRF+rerank retrieval pipeline."""

from __future__ import annotations

import math
from datetime import date

from financial_agent.knowledge.bm25 import BM25Index
from financial_agent.knowledge.dense import DenseIndex
from financial_agent.knowledge.fusion import reciprocal_rank_fusion
from financial_agent.knowledge.models import (
    AnnouncementPolicyMetadata,
    BusinessSearchInput,
    Chunk,
    Evidence,
    FAQMetadata,
    RegulatorySearchInput,
    ResearchReportMetadata,
    ResearchSearchInput,
)
from financial_agent.knowledge.policy import ResearchRetrievalPolicy, TopKPolicy
from financial_agent.knowledge.providers import EmbeddingProvider, ProviderResponseError, RerankerProvider


DEFAULT_POLICY = TopKPolicy(candidate_top_k=12, final_top_k=5)


class HybridKnowledgeRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        dense_index: DenseIndex,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        *,
        rrf_k: int = 60,
    ) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = BM25Index(chunks)
        self._dense = dense_index
        self._embedding_provider = embedding_provider
        self._reranker_provider = reranker_provider
        self._rrf_k = rrf_k
        companies = {
            chunk.metadata.company
            for chunk in chunks
            if isinstance(chunk.metadata, ResearchReportMetadata)
        }
        self.research_policy = ResearchRetrievalPolicy(companies)

    def search_research(self, request: ResearchSearchInput) -> list[Evidence]:
        eligible = {
            chunk.chunk_id for chunk in self._chunks.values()
            if _research_matches(chunk, request)
        }
        return self._search(request.query, eligible, self.research_policy.top_k(request.query))

    def search_regulatory(self, request: RegulatorySearchInput) -> list[Evidence]:
        eligible = {
            chunk.chunk_id for chunk in self._chunks.values()
            if _regulatory_matches(chunk, request)
        }
        return self._search(request.query, eligible, DEFAULT_POLICY)

    def search_business(self, request: BusinessSearchInput) -> list[Evidence]:
        eligible = {
            chunk.chunk_id for chunk in self._chunks.values()
            if _business_matches(chunk, request)
        }
        return self._search(request.query, eligible, DEFAULT_POLICY)

    def _search(self, query: str, eligible: set[str], policy: TopKPolicy) -> list[Evidence]:
        if not eligible:
            return []
        limit = min(policy.candidate_top_k, len(eligible))
        lexical = self._bm25.search(query, allowed_ids=eligible, top_k=limit)
        query_vector = self._embedding_provider.embed_query(query)
        semantic = self._dense.search(query_vector, allowed_ids=eligible, top_k=limit)
        fused = reciprocal_rank_fusion([lexical, semantic], rrf_k=self._rrf_k, top_k=limit)
        candidates = [self._chunks[item.chunk_id] for item in fused]
        reranked = self._reranker_provider.rerank(
            query,
            [f"{chunk.title}\n{chunk.content}" for chunk in candidates],
            top_n=min(policy.final_top_k, len(candidates)),
        )
        seen: set[int] = set()
        ordered = sorted(reranked, key=lambda result: (-result.score, result.index))
        evidence: list[Evidence] = []
        for result in ordered:
            if (
                result.index < 0
                or result.index >= len(candidates)
                or result.index in seen
                or not math.isfinite(result.score)
            ):
                raise ProviderResponseError("rerank", "Reranker returned invalid results")
            seen.add(result.index)
            chunk = candidates[result.index]
            evidence.append(Evidence(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_type=chunk.source_type,
                title=chunk.title,
                content=chunk.content,
                score=result.score,
                metadata=chunk.metadata,
            ))
            if len(evidence) == policy.final_top_k:
                break
        return evidence


def _research_matches(chunk: Chunk, request: ResearchSearchInput) -> bool:
    metadata = chunk.metadata
    if not isinstance(metadata, ResearchReportMetadata):
        return False
    if request.companies and metadata.company.casefold() not in _normalized(request.companies):
        return False
    if request.brokers and metadata.broker.casefold() not in _normalized(request.brokers):
        return False
    return request.as_of is None or metadata.publish_date <= request.as_of


def _regulatory_matches(chunk: Chunk, request: RegulatorySearchInput) -> bool:
    metadata = chunk.metadata
    if not isinstance(metadata, AnnouncementPolicyMetadata):
        return False
    if request.issuer is not None and metadata.issuer.casefold() != request.issuer.casefold():
        return False
    if request.as_of is None:
        return True
    return metadata.publish_date <= request.as_of and metadata.effective_date <= request.as_of


def _business_matches(chunk: Chunk, request: BusinessSearchInput) -> bool:
    metadata = chunk.metadata
    if not isinstance(metadata, FAQMetadata):
        return False
    if request.category is not None and metadata.category != request.category:
        return False
    return request.as_of is None or metadata.effective_date <= request.as_of


def _normalized(values: list[str]) -> set[str]:
    return {value.casefold() for value in values}
