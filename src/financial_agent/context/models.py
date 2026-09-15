"""Public contracts for deterministic history context selection."""

from typing import Literal

from pydantic import Field

from financial_agent.schemas import Schema, UserQuery


ContextStrategy = Literal["full_history", "last_n", "budgeted_selection"]
ContextComponent = Literal["planner", "writer", "verifier"]


class ContextPolicy(Schema):
    strategy: ContextStrategy = "full_history"
    budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    last_n: int = Field(default=6, ge=0, le=100_000)


class ContextMetrics(Schema):
    component: ContextComponent
    strategy: ContextStrategy
    budget_tokens: int = Field(ge=0)
    original_tokens: int = Field(ge=0)
    selected_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    selected_message_count: int = Field(ge=0)
    dropped_message_count: int = Field(ge=0)


class ContextSelection(Schema):
    request: UserQuery
    metrics: ContextMetrics
