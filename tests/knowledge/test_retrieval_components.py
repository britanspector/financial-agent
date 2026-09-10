from pathlib import Path

import numpy as np

from financial_agent.knowledge.bm25 import BM25Index, ScoredChunk
from financial_agent.knowledge.dense import DenseIndex
from financial_agent.knowledge.fusion import reciprocal_rank_fusion
from financial_agent.knowledge.index import EmbeddingIndexStore
from financial_agent.knowledge.ingestion import KnowledgeIngestor
from financial_agent.knowledge.models import Chunk, FAQMetadata
from financial_agent.knowledge.providers import EmbeddingDescriptor
from financial_agent.knowledge.tokenization import ChineseTokenizer


CORPUS = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "manifest.json"


class FixedEmbeddingProvider:
    def __init__(self, dimension=3, model="fixed-v1"):
        self._descriptor = EmbeddingDescriptor(model, dimension)

    @property
    def descriptor(self):
        return self._descriptor

    def embed_documents(self, texts):
        vectors = np.zeros((len(texts), self.descriptor.dimension), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % self.descriptor.dimension] = 1.0
        return vectors

    def embed_query(self, text):
        del text
        return np.eye(1, self.descriptor.dimension, dtype=np.float32)[0]


def _faq(document_id: str, title: str, content: str) -> Chunk:
    metadata = FAQMetadata(
        document_id=document_id,
        source_type="faq",
        title=title,
        category="trading",
        version="1.0",
        effective_date="2025-01-01",
    )
    return Chunk(
        chunk_id=f"{document_id}::chunk-0000",
        document_id=document_id,
        source_type="faq",
        title=title,
        content=content,
        ordinal=0,
        metadata=metadata,
    )


def test_chinese_tokenizer_is_not_whitespace_based():
    tokens = ChineseTokenizer().tokenize("融资保证金比例调整")

    assert len(tokens) > 1
    assert "保证金" in tokens


def test_bm25_finds_chinese_terms_without_spaces():
    chunks = [
        _faq("FAQ-1", "融资保证金", "最低保证金比例调整为百分之九十"),
        _faq("FAQ-2", "密码重置", "交易密码连续输错后可以重置"),
    ]

    result = BM25Index(chunks).search("保证金比例", top_k=1)

    assert result[0].chunk_id == "FAQ-1::chunk-0000"


def test_dense_index_uses_cosine_similarity_and_filter():
    index = DenseIndex(["a", "b", "c"], np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32))

    assert [item.chunk_id for item in index.search(np.array([1, 0]), top_k=2)] == ["a", "b"]
    assert [item.chunk_id for item in index.search(np.array([1, 0]), allowed_ids={"b", "c"}, top_k=1)] == ["b"]


def test_rrf_combines_rankings_without_raw_score_mixing():
    first = [ScoredChunk("a", 100.0), ScoredChunk("b", 1.0)]
    second = [ScoredChunk("b", 0.9), ScoredChunk("c", 0.8)]

    fused = reciprocal_rank_fusion([first, second], rrf_k=60)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]


def test_embedding_index_persists_and_detects_staleness(tmp_path):
    chunks = KnowledgeIngestor().ingest(CORPUS)[:4]
    provider = FixedEmbeddingProvider()
    store = EmbeddingIndexStore(tmp_path / "vectors.npz")

    built = store.build(chunks, provider, batch_size=2)
    assert built.dimension == 3
    assert store.state(chunks, provider.descriptor) == "ready"
    assert store.load(chunks, provider.descriptor).dimension == 3

    changed = list(chunks)
    changed[0] = changed[0].model_copy(update={"content": changed[0].content + " changed"})
    assert store.state(changed, provider.descriptor) == "stale"
    assert store.state(chunks, EmbeddingDescriptor("other-model", 3)) == "stale"
