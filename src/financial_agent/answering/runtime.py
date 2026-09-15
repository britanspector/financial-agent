from financial_agent.answering.providers import AnswerProvider
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.answering.service import AnswerWriter
from financial_agent.config import Settings
from financial_agent.context import (
    ContextManager, HistoryRetriever, SummaryProvider, TokenEstimator, build_context_manager,
    context_policy_from_settings,
)
from financial_agent.verifier.evidence import OutputCatalog


def build_answer_writer(
    settings: Settings,
    catalog: OutputCatalog,
    *,
    provider: AnswerProvider | None = None,
    context_manager: ContextManager | None = None,
    token_estimator: TokenEstimator | None = None,
    summary_provider: SummaryProvider | None = None,
    history_retriever: HistoryRetriever | None = None,
) -> AnswerWriter:
    if context_manager is not None and (
        token_estimator is not None or summary_provider is not None or history_retriever is not None
    ):
        raise ValueError("Pass context_manager or context dependencies, not both")
    if provider is None:
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required")
        provider = QwenAnswerProvider(settings.qwen_api_key.get_secret_value(), model=settings.answer_model,
                                      base_url=settings.answer_base_url, timeout=settings.answer_timeout_seconds,
                                      temperature=settings.answer_temperature)
    return AnswerWriter(
        provider,
        catalog,
        context_manager=context_manager or build_context_manager(
        token_estimator,
        settings=settings,
        summary_provider=summary_provider,
        history_retriever=history_retriever,
        ),
        context_policy=context_policy_from_settings(settings, "writer"),
    )
