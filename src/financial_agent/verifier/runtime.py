"""Verifier composition helpers."""

from financial_agent.config import Settings
from financial_agent.context import (
    ContextManager, SummaryProvider, TokenEstimator, build_context_manager, context_policy_from_settings,
)
from financial_agent.verifier.providers import VerifierProvider
from financial_agent.verifier.qwen_provider import QwenVerifierProvider
from financial_agent.verifier.service import OutputCatalog, StructuredVerifier


def build_verifier(
    settings: Settings,
    catalog: OutputCatalog,
    *,
    provider: VerifierProvider | None = None,
    context_manager: ContextManager | None = None,
    token_estimator: TokenEstimator | None = None,
    summary_provider: SummaryProvider | None = None,
) -> StructuredVerifier:
    if context_manager is not None and (token_estimator is not None or summary_provider is not None):
        raise ValueError("Pass context_manager or context dependencies, not both")
    if provider is None:
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required")
        provider = QwenVerifierProvider(
            settings.qwen_api_key.get_secret_value(),
            model=settings.verifier_model,
            base_url=settings.verifier_base_url,
            timeout=settings.verifier_timeout_seconds,
            temperature=settings.verifier_temperature,
        )
    return StructuredVerifier(
        provider,
        catalog,
        context_manager=context_manager or build_context_manager(
            token_estimator, settings=settings, summary_provider=summary_provider,
        ),
        context_policy=context_policy_from_settings(settings, "verifier"),
    )
