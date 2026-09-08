from types import SimpleNamespace

import pytest

from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CallContext, Credential, CredentialStore
from financial_agent.user_data.fixtures import seed_user_data
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.service import UserDataService


class SpyRepository:
    def __init__(self, repository):
        self.repository = repository
        self.calls = []

    def get_profile(self, user_id):
        self.calls.append(("profile", user_id))
        return self.repository.get_profile(user_id)

    def get_portfolio(self, user_id):
        self.calls.append(("portfolio", user_id))
        return self.repository.get_portfolio(user_id)

    def get_transactions(self, query):
        self.calls.append(("transactions", query.user_id))
        return self.repository.get_transactions(query)


@pytest.fixture
def db_path(isolated_settings, tmp_path):
    return seed_user_data(tmp_path / "user_data.db")


@pytest.fixture
def repository(db_path):
    return SQLiteUserDataRepository(db_path)


@pytest.fixture
def credentials():
    return CredentialStore([
        Credential(api_key="test-only-full-key", principal_id="synthetic-full",
                   user_ids={f"syn-user-{i:03}" for i in range(1, 7)} | {"syn-user-999"},
                   scopes={"read:profile", "read:portfolio"}),
        Credential(api_key="test-only-profile-key", principal_id="synthetic-profile",
                   user_ids={"syn-user-001"}, scopes={"read:profile"}),
        Credential(api_key="test-only-portfolio-key", principal_id="synthetic-portfolio",
                   user_ids={"syn-user-001"}, scopes={"read:portfolio"}),
        Credential(api_key="test-only-no-scope", principal_id="synthetic-no-scope",
                   user_ids={"syn-user-001"}, scopes=set()),
        Credential(api_key="test-only-other-user", principal_id="synthetic-other-user",
                   user_ids={"syn-user-002"}, scopes={"read:profile", "read:portfolio"}),
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
