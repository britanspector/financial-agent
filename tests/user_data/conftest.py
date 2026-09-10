from types import SimpleNamespace

import pytest

from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CallContext, Credential, CredentialStore
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.synthetic_generator import generate_synthetic_data
from financial_agent.user_data.service import UserDataService


class SpyRepository:
    def __init__(self, repository):
        self.repository = repository
        self.calls = []

    def get_customer_context(self, user_id):
        self.calls.append(("customer_context", user_id))
        return self.repository.get_customer_context(user_id)

    def get_margin_account(self, query):
        self.calls.append(("margin_account", query.user_id))
        return self.repository.get_margin_account(query)

    def get_portfolio_positions(self, user_id):
        self.calls.append(("portfolio_positions", user_id))
        return self.repository.get_portfolio_positions(user_id)

    def get_portfolio_analytics(self, user_id):
        self.calls.append(("portfolio_analytics", user_id))
        return self.repository.get_portfolio_analytics(user_id)


@pytest.fixture
def db_path(isolated_settings, tmp_path):
    return generate_synthetic_data(tmp_path / "user_data.db", seed=20260910, user_count=20).path


@pytest.fixture
def repository(db_path):
    return SQLiteUserDataRepository(db_path)


@pytest.fixture
def credentials():
    return CredentialStore([
        Credential(api_key="test-only-full-key", principal_id="synthetic-full",
                   user_ids={f"syn-user-{i:04}" for i in range(1, 21)} | {"syn-user-999"},
                   scopes={"read:customer_context", "read:margin_account", "read:portfolio_positions", "read:portfolio_analytics"}),
        Credential(api_key="test-only-context-key", principal_id="synthetic-context",
                   user_ids={"syn-user-0001"}, scopes={"read:customer_context"}),
        Credential(api_key="test-only-margin-key", principal_id="synthetic-margin",
                   user_ids={"syn-user-0001"}, scopes={"read:margin_account"}),
        Credential(api_key="test-only-positions-key", principal_id="synthetic-positions",
                   user_ids={"syn-user-0001"}, scopes={"read:portfolio_positions"}),
        Credential(api_key="test-only-analytics-key", principal_id="synthetic-analytics",
                   user_ids={"syn-user-0001"}, scopes={"read:portfolio_analytics"}),
        Credential(api_key="test-only-no-scope", principal_id="synthetic-no-scope",
                   user_ids={"syn-user-001"}, scopes=set()),
        Credential(api_key="test-only-other-user", principal_id="synthetic-other-user",
                   user_ids={"syn-user-0002"}, scopes={"read:customer_context", "read:margin_account", "read:portfolio_positions", "read:portfolio_analytics"}),
    ])


@pytest.fixture
def make_system(repository, credentials, tmp_path):
    def make(*, fault=None, audit=None, clock=None, repo=None):
        spy = SpyRepository(repo if repo is not None else repository)
        audit_path = tmp_path / "audit.jsonl"
        sink = audit if audit is not None else JsonlAuditSink(audit_path)
        options = {"before_read": fault}
        if clock is not None:
            options["clock"] = clock
        service = UserDataService(spy, credentials, sink, **options)
        return SimpleNamespace(
            registry=register_user_tools(service), repository=spy, audit_path=audit_path,
            context=CallContext(api_key="test-only-full-key"),
        )
    return make
