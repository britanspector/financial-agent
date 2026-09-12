"""Composition root for the independent market-data tools."""

from financial_agent.config import Settings
from financial_agent.market_data.tushare_provider import TushareProvider
from financial_agent.market_data.models import (
    MarketHistory, MarketHistoryInput, MarketSnapshot, MarketSnapshotInput,
)
from financial_agent.market_data.service import MarketDataService
from financial_agent.tools.registry import ToolRegistry, ToolSpec


def register_market_tools(service: MarketDataService) -> ToolRegistry:
    registry = ToolRegistry(service)
    registry.register(ToolSpec(
        "get_market_snapshot", "Read the latest available A-share stock daily close; not real-time, index, news, FX, or prediction data",
        MarketSnapshotInput, MarketSnapshot, None, "market_snapshot",
    ))
    registry.register(ToolSpec(
        "get_market_history", "Read unadjusted A-share stock daily historical prices; not index, real-time, news, FX, or prediction data",
        MarketHistoryInput, MarketHistory, None, "market_history",
    ))
    return registry


def build_market_tools(settings: Settings | None = None) -> ToolRegistry:
    settings = settings or Settings()
    if settings.tushare_token is None:
        raise ValueError("Tushare token is required")
    provider = TushareProvider(
        settings.tushare_token.get_secret_value(),
        base_url=settings.tushare_base_url,
        timeout=settings.market_data_timeout_seconds,
    )
    return register_market_tools(MarketDataService(provider))
