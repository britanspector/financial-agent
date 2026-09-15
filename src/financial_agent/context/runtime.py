"""Context Manager composition helpers."""

from __future__ import annotations

from financial_agent.config import Settings
from financial_agent.context.manager import ContextManager
from financial_agent.context.models import ContextComponent, ContextPolicy
from financial_agent.context.token_estimation import TokenEstimator


def build_context_manager(estimator: TokenEstimator | None = None) -> ContextManager:
    return ContextManager(estimator)


def context_policy_from_settings(settings: Settings, component: ContextComponent) -> ContextPolicy:
    budgets = {
        "planner": settings.planner_context_budget_tokens,
        "writer": settings.answer_context_budget_tokens,
        "verifier": settings.verifier_context_budget_tokens,
    }
    return ContextPolicy(
        strategy=settings.context_strategy,
        budget_tokens=budgets[component],
        last_n=settings.context_last_n,
    )
