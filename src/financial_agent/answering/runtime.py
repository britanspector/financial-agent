from financial_agent.answering.providers import AnswerProvider
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.answering.service import AnswerWriter
from financial_agent.config import Settings
from financial_agent.verifier.evidence import OutputCatalog


def build_answer_writer(settings: Settings, catalog: OutputCatalog, *, provider: AnswerProvider | None = None) -> AnswerWriter:
    if provider is None:
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required")
        provider = QwenAnswerProvider(settings.qwen_api_key.get_secret_value(), model=settings.answer_model,
                                      base_url=settings.answer_base_url, timeout=settings.answer_timeout_seconds,
                                      temperature=settings.answer_temperature)
    return AnswerWriter(provider, catalog)
