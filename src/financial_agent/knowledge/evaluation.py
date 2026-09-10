"""Minimal retrieval metrics; evaluation cases live outside the corpus."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field

from financial_agent.knowledge.models import Evidence
from financial_agent.schemas import Schema


class RetrievalEvalCase(Schema):
    case_id: str = Field(min_length=1)
    tool: Literal[
        "search_research_reports",
        "search_regulatory_knowledge",
        "search_business_knowledge",
    ]
    arguments: dict[str, Any]
    relevant_document_ids: list[str] = Field(min_length=1)


class RetrievalMetrics(Schema):
    queries: int = Field(ge=1)
    hit_at_1: float = Field(ge=0, le=1)
    hit_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)


def evaluate_retrieval(
    cases: list[RetrievalEvalCase],
    search: Callable[[RetrievalEvalCase], list[Evidence]],
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("evaluation requires at least one case")
    hit_1 = hit_5 = 0
    reciprocal_rank = recall = 0.0
    for case in cases:
        results = search(case)
        ranked_ids = [item.document_id for item in results]
        relevant = set(case.relevant_document_ids)
        hit_1 += bool(ranked_ids and ranked_ids[0] in relevant)
        hit_5 += bool(relevant.intersection(ranked_ids[:5]))
        first_rank = next((rank for rank, item in enumerate(ranked_ids, start=1) if item in relevant), None)
        reciprocal_rank += 1.0 / first_rank if first_rank is not None else 0.0
        recall += len(relevant.intersection(ranked_ids[:5])) / len(relevant)
    count = len(cases)
    return RetrievalMetrics(
        queries=count,
        hit_at_1=hit_1 / count,
        hit_at_5=hit_5 / count,
        mrr=reciprocal_rank / count,
        recall_at_5=recall / count,
    )
