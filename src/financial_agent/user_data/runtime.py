"""Composition root: choose adapters here, outside tool execution."""

from financial_agent.config import Settings
from financial_agent.tools.registry import ToolRegistry, ToolSpec
from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CredentialStore
from financial_agent.user_data.models import Portfolio, Profile, TransactionPage, TransactionsInput, UserInput
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.service import UserDataService


def register_user_tools(service: UserDataService) -> ToolRegistry:
    registry = ToolRegistry(service)
    for spec in (
        ToolSpec("get_user_profile", "Read a synthetic user profile", UserInput, Profile, "read:profile", "profile"),
        ToolSpec("get_user_portfolio", "Read synthetic accounts and holdings", UserInput, Portfolio, "read:portfolio", "portfolio"),
        ToolSpec("get_user_transactions", "Read synthetic transactions in time order", TransactionsInput, TransactionPage, "read:portfolio", "transactions"),
    ):
        registry.register(spec)
    return registry


def build_user_tools(settings: Settings) -> ToolRegistry:
    return register_user_tools(UserDataService(
        repository=SQLiteUserDataRepository(settings.user_db_path),
        credentials=CredentialStore.from_json(settings.user_api_keys),
        audit=JsonlAuditSink(settings.audit_path),
    ))
