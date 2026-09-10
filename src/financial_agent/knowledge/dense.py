"""NumPy cosine-similarity retrieval."""

from __future__ import annotations

import numpy as np

from financial_agent.knowledge.bm25 import ScoredChunk


class DenseIndex:
    def __init__(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
            raise ValueError("embedding matrix must align with chunk_ids")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding matrix contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._vectors = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        self._chunk_ids = list(chunk_ids)

    @property
    def dimension(self) -> int:
        return self._vectors.shape[1]

    def search(
        self,
        query_vector: np.ndarray,
        *,
        allowed_ids: set[str] | None = None,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError("query embedding dimension does not match index")
        if not np.isfinite(vector).all():
            raise ValueError("query embedding contains non-finite values")
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm else np.zeros_like(vector)
        similarities = self._vectors @ normalized
        scored = [
            ScoredChunk(chunk_id, float(similarities[index]))
            for index, chunk_id in enumerate(self._chunk_ids)
            if allowed_ids is None or chunk_id in allowed_ids
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:max(top_k, 0)]
