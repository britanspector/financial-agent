"""Planner composition helpers."""

from financial_agent.config import Settings
from financial_agent.context import ContextManager, TokenEstimator, context_policy_from_settings
from financial_agent.planner.providers import PlannerProvider
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.planner.service import StructuredPlanner, ToolCatalog
from financial_agent.planner.validator import PlanValidator


def build_planner(
    settings: Settings,
    catalog: ToolCatalog,
    *,
    provider: PlannerProvider | None = None,
    context_manager: ContextManager | None = None,
    token_estimator: TokenEstimator | None = None,
) -> tuple[StructuredPlanner, PlanValidator]:
    if context_manager is not None and token_estimator is not None:
        raise ValueError("Pass context_manager or token_estimator, not both")
    if provider is None:
        if settings.qwen_api_key is None:
            raise ValueError("Qwen API key is required")
        provider = QwenPlannerProvider(
            settings.qwen_api_key.get_secret_value(),
            model=settings.planner_model,
            base_url=settings.planner_base_url,
            timeout=settings.planner_timeout_seconds,
            temperature=settings.planner_temperature,
        )
    manager = context_manager or ContextManager(token_estimator)
    return (
        StructuredPlanner(
            provider,
            catalog,
            context_manager=manager,
            context_policy=context_policy_from_settings(settings, "planner"),
        ),
        PlanValidator(catalog, max_tasks=settings.planner_max_tasks),
    )
