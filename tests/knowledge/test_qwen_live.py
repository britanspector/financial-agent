import numpy as np
import pytest

from financial_agent.config import Settings
from financial_agent.knowledge.qwen_providers import QwenEmbeddingProvider, QwenRerankerProvider


pytestmark = pytest.mark.live


def settings_or_skip():
    settings = Settings()
    if settings.qwen_api_key is None:
        pytest.skip("FINANCIAL_AGENT_QWEN_API_KEY is not configured")
    return settings


@pytest.mark.parametrize("model", ["qwen3.7-text-embedding", "qwen3.7-text-embedding-flash"])
def test_qwen_embedding_live(model):
    settings = settings_or_skip()
    provider = QwenEmbeddingProvider(
        settings.qwen_api_key.get_secret_value(),
        model=model,
        dimension=1024,
        base_url=settings.qwen_embedding_base_url,
        timeout=settings.qwen_timeout_seconds,
    )

    vectors = provider.embed_documents(["融资保证金比例", "企业软件订阅收入"])
    query = provider.embed_query("两融保证金")

    assert vectors.shape == (2, 1024)
    assert query.shape == (1024,)
    assert np.isfinite(vectors).all()


def test_qwen_reranker_live():
    settings = settings_or_skip()
    provider = QwenRerankerProvider(
        settings.qwen_api_key.get_secret_value(),
        model=settings.qwen_reranker_model,
        base_url=settings.qwen_reranker_base_url,
        timeout=settings.qwen_timeout_seconds,
    )

    results = provider.rerank(
        "融资保证金比例",
        ["融资买入最低保证金比例调整为90%", "企业软件订阅收入增长"],
        top_n=2,
    )

    assert len(results) == 2
    assert results[0].index == 0
    assert results[0].score >= results[1].score
