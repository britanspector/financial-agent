"""Public contracts for deterministic history context selection."""

from typing import Literal

from pydantic import Field

from financial_agent.schemas import Schema, UserQuery


ContextStrategy = Literal["full_history", "last_n", "budgeted_selection", "summary_compression"]
ContextComponent = Literal["planner", "writer", "verifier"]


class ContextPolicy(Schema):
    strategy: ContextStrategy = "full_history"
    budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    last_n: int = Field(default=6, ge=0, le=100_000)
    summary_recent_n: int = Field(default=3, ge=0, le=100_000)
    summary_budget_ratio: float = Field(default=0.4, ge=0, le=1)


class SummaryFact(Schema):
    category: Literal["entity", "time_range", "constraint", "confirmed_intent", "planning_fact"]
    content: str = Field(min_length=1)
    source_message_index: int = Field(ge=0)


class HistorySummary(Schema):
    facts: list[SummaryFact] = Field(default_factory=list)


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


class ContextSelection(Schema):
    request: UserQuery
    metrics: ContextMetrics
    summary: HistorySummary | None = None
