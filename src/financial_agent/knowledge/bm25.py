"""Small in-memory Okapi BM25 implementation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from financial_agent.knowledge.models import Chunk
from financial_agent.knowledge.tokenization import ChineseTokenizer


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenizer: ChineseTokenizer | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer = tokenizer or ChineseTokenizer()
        self._k1 = k1
        self._b = b
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._term_frequencies = [Counter(self._tokenizer.tokenize(_search_text(chunk))) for chunk in chunks]
        self._lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(chunks)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, allowed_ids: set[str] | None = None, top_k: int = 10) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        query_terms = self._tokenizer.tokenize(query)
        scores: list[ScoredChunk] = []
        for index, chunk_id in enumerate(self._chunk_ids):
            if allowed_ids is not None and chunk_id not in allowed_ids:
                continue
            score = self._score(query_terms, index)
            scores.append(ScoredChunk(chunk_id, score))
        scores.sort(key=lambda item: (-item.score, item.chunk_id))
        return scores[:top_k]

    def _score(self, query_terms: list[str], index: int) -> float:
        frequencies = self._term_frequencies[index]
        length = self._lengths[index]
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self._k1 * (
                1.0 - self._b + self._b * length / self._average_length
            ) if self._average_length else frequency
            score += self._idf.get(term, 0.0) * frequency * (self._k1 + 1.0) / denominator * query_frequency
        return score


def _search_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.content}"
