"""Verifier composition helpers."""

from financial_agent.config import Settings
from financial_agent.verifier.providers import VerifierProvider
from financial_agent.verifier.qwen_provider import QwenVerifierProvider
from financial_agent.verifier.service import OutputCatalog, StructuredVerifier


def build_verifier(
    settings: Settings,
    catalog: OutputCatalog,
    *,
    provider: VerifierProvider | None = None,
) -> StructuredVerifier:
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
    return StructuredVerifier(provider, catalog)
