"""Retrieval policies kept separate from ranking implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKPolicy:
    candidate_top_k: int
    final_top_k: int


class ResearchRetrievalPolicy:
    """Expand only the recall pool when a query names more companies."""

    def __init__(self, companies: set[str]) -> None:
        self._companies = frozenset(company for company in companies if company)

    @property
    def companies(self) -> frozenset[str]:
        return self._companies

    def entity_count(self, query: str) -> int:
        normalized = query.casefold()
        return sum(company.casefold() in normalized for company in self._companies)

    def top_k(self, query: str) -> TopKPolicy:
        count = self.entity_count(query)
        if count <= 1:
            return TopKPolicy(candidate_top_k=12, final_top_k=5)
        if count == 2:
            return TopKPolicy(candidate_top_k=20, final_top_k=8)
        return TopKPolicy(candidate_top_k=30, final_top_k=10)
