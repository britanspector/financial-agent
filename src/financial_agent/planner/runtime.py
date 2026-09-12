"""Planner composition helpers."""

from financial_agent.config import Settings
from financial_agent.planner.providers import PlannerProvider
from financial_agent.planner.qwen_provider import QwenPlannerProvider
from financial_agent.planner.service import StructuredPlanner, ToolCatalog
from financial_agent.planner.validator import PlanValidator


def build_planner(
    settings: Settings,
    catalog: ToolCatalog,
    *,
    provider: PlannerProvider | None = None,
) -> tuple[StructuredPlanner, PlanValidator]:
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
    return StructuredPlanner(provider, catalog), PlanValidator(catalog, max_tasks=settings.planner_max_tasks)
