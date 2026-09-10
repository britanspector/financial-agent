"""Rank-only reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from financial_agent.knowledge.bm25 import ScoredChunk


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    score: float


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], *, rrf_k: int = 60, top_k: int | None = None
) -> list[FusedChunk]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(item.chunk_id)
            if item.chunk_id not in first_seen:
                first_seen[item.chunk_id] = sequence
                sequence += 1
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = [FusedChunk(chunk_id, score) for chunk_id, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, first_seen[item.chunk_id], item.chunk_id))
    return fused if top_k is None else fused[:max(top_k, 0)]
