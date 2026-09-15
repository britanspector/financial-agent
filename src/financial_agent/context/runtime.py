"""Context Manager composition helpers."""

from __future__ import annotations

from financial_agent.config import Settings
from financial_agent.context.manager import ContextManager
from financial_agent.context.models import ContextComponent, ContextPolicy
from financial_agent.context.qwen_summary_provider import QwenSummaryProvider
from financial_agent.context.retrieval import HistoryRetriever, LexicalHistoryRetriever
from financial_agent.context.summarizer import HistorySummarizer
from financial_agent.context.summary_providers import SummaryProvider
from financial_agent.context.token_estimation import TokenEstimator


def build_context_manager(
    estimator: TokenEstimator | None = None,
    *,
    settings: Settings | None = None,
    summary_provider: SummaryProvider | None = None,
    history_retriever: HistoryRetriever | None = None,
) -> ContextManager:
    summarizer = None
    cache_size = settings.context_summary_cache_size if settings else 128
    if summary_provider is not None:
        summarizer = HistorySummarizer(
            summary_provider,
            max_facts=settings.context_summary_max_facts if settings else 24,
        )
    elif settings is not None and settings.context_strategy in {"summary_compression", "summary_retrieval"}:
        if settings.qwen_api_key is None:
            raise ValueError(f"Qwen API key is required for {settings.context_strategy}")
        provider = QwenSummaryProvider(
            settings.qwen_api_key.get_secret_value(),
            model=settings.summary_model,
            base_url=settings.summary_base_url,
            timeout=settings.summary_timeout_seconds,
            temperature=settings.summary_temperature,
        )
        summarizer = HistorySummarizer(provider, max_facts=settings.context_summary_max_facts)
    return ContextManager(
        estimator,
        summarizer=summarizer,
        retriever=history_retriever or LexicalHistoryRetriever(),
        summary_cache_size=cache_size,
    )


def context_policy_from_settings(settings: Settings, component: ContextComponent) -> ContextPolicy:
    budgets = {
        "planner": settings.planner_context_budget_tokens,
        "writer": settings.answer_context_budget_tokens,
        "verifier": settings.verifier_context_budget_tokens,
    }
    return ContextPolicy(
        strategy=settings.context_strategy,
        budget_tokens=budgets[component],
        last_n=settings.context_last_n,
        summary_recent_n=settings.context_summary_recent_n,
        summary_budget_ratio=settings.context_summary_budget_ratio,
        retrieval_top_k=settings.context_retrieval_top_k,
        retrieval_min_score=settings.context_retrieval_min_score,
        retrieval_recent_reservation_ratio=settings.context_retrieval_recent_reservation_ratio,
        retrieval_protected_summary_reservation_ratio=(
            settings.context_retrieval_protected_summary_reservation_ratio
        ),
        retrieval_history_reservation_ratio=settings.context_retrieval_history_reservation_ratio,
    )
