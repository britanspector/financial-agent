"""Composition root for all nine Phase 1 tools."""

from __future__ import annotations

from financial_agent.config import Settings
from financial_agent.knowledge.providers import EmbeddingProvider, RerankerProvider
from financial_agent.knowledge.runtime import build_rag_tools
from financial_agent.market_data.runtime import build_market_tools
from financial_agent.tools.composite import CompositeToolRegistry, merge_registries
from financial_agent.user_data.runtime import build_user_tools


def build_agent_tools(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
) -> CompositeToolRegistry:
    """Build and combine User, Market, and RAG registries without changing them."""
    return merge_registries(
        build_user_tools(settings),
        build_market_tools(settings),
        build_rag_tools(
            settings,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
        ),
    )
