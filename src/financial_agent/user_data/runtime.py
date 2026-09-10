"""Composition root: choose adapters here, outside tool execution."""

from financial_agent.config import Settings
from financial_agent.tools.registry import ToolRegistry, ToolSpec
from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CredentialStore
from financial_agent.user_data.models import (
    CustomerContext, MarginAccount, MarginAccountInput, PortfolioAnalytics, PortfolioPositions, UserInput,
)
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.service import UserDataService
from financial_agent.user_data.http_client import HttpUserDataClient


def register_user_tools(service) -> ToolRegistry:
    registry = ToolRegistry(service)
    for spec in (
        ToolSpec("get_customer_context", "Read customer identity, usage, and business permissions", UserInput, CustomerContext, "read:customer_context", "customer_context"),
        ToolSpec("get_margin_account", "Read the current margin account and daily history", MarginAccountInput, MarginAccount, "read:margin_account", "margin_account"),
        ToolSpec("get_portfolio_positions", "Read industry and stock positions", UserInput, PortfolioPositions, "read:portfolio_positions", "portfolio_positions"),
        ToolSpec("get_portfolio_analytics", "Read returns, risk, factors, and peer ranking", UserInput, PortfolioAnalytics, "read:portfolio_analytics", "portfolio_analytics"),
    ):
        registry.register(spec)
    return registry


def build_local_user_tools(settings: Settings) -> ToolRegistry:
    return register_user_tools(UserDataService(
        repository=SQLiteUserDataRepository(settings.user_db_path),
        credentials=CredentialStore.from_json(settings.user_api_keys),
        audit=JsonlAuditSink(settings.audit_path),
    ))


def build_user_tools(settings: Settings) -> ToolRegistry:
    """Build the public Tool registry backed by the HTTP user-data service."""
    return register_user_tools(HttpUserDataClient(
        settings.user_data_base_url,
        api_key=settings.caller_api_key.get_secret_value() if settings.caller_api_key else None,
        timeout=settings.user_data_timeout_seconds,
    ))
