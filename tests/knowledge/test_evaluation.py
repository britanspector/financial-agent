import hashlib
import json
from pathlib import Path

import numpy as np

from financial_agent.knowledge.dense import DenseIndex
from financial_agent.knowledge.evaluation import RetrievalEvalCase, evaluate_retrieval
from financial_agent.knowledge.ingestion import KnowledgeIngestor
from financial_agent.knowledge.models import BusinessSearchInput, RegulatorySearchInput, ResearchSearchInput
from financial_agent.knowledge.providers import EmbeddingDescriptor, RerankResult
from financial_agent.knowledge.retrieval import HybridKnowledgeRetriever
from financial_agent.knowledge.tokenization import ChineseTokenizer


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "knowledge" / "manifest.json"
CASES = Path(__file__).with_name("eval_cases.json")


class OfflineEmbedding:
    descriptor = EmbeddingDescriptor("offline-eval-v1", 512)

    def __init__(self):
        self.tokenizer = ChineseTokenizer()

    def one(self, text):
        vector = np.zeros(self.descriptor.dimension, dtype=np.float32)
        for token in self.tokenizer.tokenize(text):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.descriptor.dimension] += 1
        return vector

    def embed_documents(self, texts):
        return np.vstack([self.one(text) for text in texts])

    def embed_query(self, text):
        return self.one(text)


class OfflineReranker:
    def __init__(self):
        self.tokenizer = ChineseTokenizer()

    def rerank(self, query, documents, *, top_n):
        query_terms = set(self.tokenizer.tokenize(query))
        results = []
        for index, document in enumerate(documents):
            document_terms = self.tokenizer.tokenize(document)
            score = sum(document_terms.count(term) for term in query_terms) / max(len(document_terms), 1)
            results.append(RerankResult(index, score))
        return sorted(results, key=lambda item: (-item.score, item.index))[:top_n]


def test_minimal_retrieval_eval():
    cases = [RetrievalEvalCase.model_validate(item) for item in json.loads(CASES.read_text(encoding="utf-8"))]
    chunks = KnowledgeIngestor().ingest(CORPUS)
    embedding = OfflineEmbedding()
    vectors = embedding.embed_documents([f"{chunk.title}\n{chunk.content}" for chunk in chunks])
    retriever = HybridKnowledgeRetriever(
        chunks,
        DenseIndex([chunk.chunk_id for chunk in chunks], vectors),
        embedding,
        OfflineReranker(),
    )

    def search(case):
        if case.tool == "search_research_reports":
            return retriever.search_research(ResearchSearchInput.model_validate(case.arguments))
        if case.tool == "search_regulatory_knowledge":
            return retriever.search_regulatory(RegulatorySearchInput.model_validate(case.arguments))
        return retriever.search_business(BusinessSearchInput.model_validate(case.arguments))

    metrics = evaluate_retrieval(cases, search)
    print(metrics.model_dump_json())

    assert metrics.queries == 24
    assert metrics.hit_at_5 >= 0.90
    assert metrics.mrr >= 0.75
