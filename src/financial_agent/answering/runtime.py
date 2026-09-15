from financial_agent.answering.providers import AnswerProvider
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.answering.service import AnswerWriter
from financial_agent.config import Settings
from financial_agent.context import ContextManager, TokenEstimator, context_policy_from_settings
from financial_agent.verifier.evidence import OutputCatalog


def build_answer_writer(
    settings: Settings,
    catalog: OutputCatalog,
    *,
    provider: AnswerProvider | None = None,
    context_manager: ContextManager | None = None,
    token_estimator: TokenEstimator | None = None,
) -> AnswerWriter:
    if context_manager is not None and token_estimator is not None:
        raise ValueError("Pass context_manager or token_estimator, not both")
    if provider is None:
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required")
        provider = QwenAnswerProvider(settings.qwen_api_key.get_secret_value(), model=settings.answer_model,
                                      base_url=settings.answer_base_url, timeout=settings.answer_timeout_seconds,
                                      temperature=settings.answer_temperature)
    return AnswerWriter(
        provider,
        catalog,
        context_manager=context_manager or ContextManager(token_estimator),
        context_policy=context_policy_from_settings(settings, "writer"),
    )
