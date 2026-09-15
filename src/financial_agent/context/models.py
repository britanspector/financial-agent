"""Public contracts for deterministic history context selection."""

from typing import Literal

from pydantic import Field, model_validator

from financial_agent.schemas import Message, Schema, UserQuery


ContextStrategy = Literal[
    "full_history", "last_n", "budgeted_selection", "summary_compression", "summary_retrieval"
]
ContextComponent = Literal["planner", "writer", "verifier"]


class ContextPolicy(Schema):
    strategy: ContextStrategy = "full_history"
    budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    last_n: int = Field(default=6, ge=0, le=100_000)
    summary_recent_n: int = Field(default=3, ge=0, le=100_000)
    summary_budget_ratio: float = Field(default=0.4, ge=0, le=1)
    retrieval_top_k: int = Field(default=4, ge=0, le=100)
    retrieval_min_score: float = Field(default=0.15, ge=0, le=2)
    retrieval_recent_reservation_ratio: float = Field(default=0.3, ge=0, le=1)
    retrieval_protected_summary_reservation_ratio: float = Field(default=0.1, ge=0, le=1)
    retrieval_history_reservation_ratio: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def valid_retrieval_reservations(self):
        total = (
            self.retrieval_recent_reservation_ratio
            + self.retrieval_protected_summary_reservation_ratio
            + self.retrieval_history_reservation_ratio
        )
        if total > 1 + 1e-9:
            raise ValueError("Context retrieval reservation ratios must sum to at most 1")
        return self


class SummaryFact(Schema):
    category: Literal["entity", "time_range", "constraint", "confirmed_intent", "planning_fact"]
    content: str = Field(min_length=1)
    source_message_index: int = Field(ge=0)


class HistorySummary(Schema):
    facts: list[SummaryFact] = Field(default_factory=list)


class RetrievedHistoryTurn(Schema):
    rank: int = Field(ge=1)
    score: float = Field(ge=0, allow_inf_nan=False)
    turn_index: int = Field(ge=0)
    message_indexes: list[int] = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)


class ContextMetrics(Schema):
    component: ContextComponent
    strategy: ContextStrategy
    budget_tokens: int = Field(ge=0)
    original_tokens: int = Field(ge=0)
    selected_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    selected_message_count: int = Field(ge=0)
    dropped_message_count: int = Field(ge=0)
    summary_used: bool = False
    summary_cache_hit: bool = False
    summarized_message_count: int = Field(default=0, ge=0)
    raw_message_count: int = Field(default=0, ge=0)
    summary_fact_count: int = Field(default=0, ge=0)
    summary_tokens: int = Field(default=0, ge=0)
    summary_fallback_reason: Literal[
        "provider_error", "invalid_summary", "empty_summary", "protected_fact_missing", "over_budget"
    ] | None = None
    retrieval_active: bool = False
    retrieval_candidate_turn_count: int = Field(default=0, ge=0)
    retrieval_eligible_turn_count: int = Field(default=0, ge=0)
    retrieved_turn_count: int = Field(default=0, ge=0)
    retrieved_message_count: int = Field(default=0, ge=0)
    retrieved_tokens: int = Field(default=0, ge=0)
    recent_message_count: int = Field(default=0, ge=0)
    retrieval_fallback_reason: Literal[
        "retriever_error", "summary_error", "protected_fact_over_budget", "both_failed"
    ] | None = None
    protected_fact_count: int = Field(default=0, ge=0)
    protected_fact_covered_count: int = Field(default=0, ge=0)


class ContextSelection(Schema):
    request: UserQuery
    metrics: ContextMetrics
    summary: HistorySummary | None = None
    retrieved_history: list[RetrievedHistoryTurn] = Field(default_factory=list)
