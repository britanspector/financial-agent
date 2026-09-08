from decimal import Decimal
import sqlite3

import pytest

from financial_agent.user_data.fixtures import seed_user_data
from financial_agent.user_data.models import TransactionsInput
from financial_agent.user_data.repository import RepositoryUnavailable, SQLiteUserDataRepository, UserNotFound


def dump(path):
    connection = sqlite3.connect(path)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def test_fixtures_are_repeatable_and_never_silently_overwritten(db_path, tmp_path):
    original = dump(db_path)
    assert dump(seed_user_data(tmp_path / "second.db")) == original
    with pytest.raises(FileExistsError):
        seed_user_data(db_path)
    assert dump(db_path) == original
    seed_user_data(db_path, overwrite=True)
    assert dump(db_path) == original


def test_foreign_keys_and_decimal_storage(db_path):
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 6
        assert connection.execute("SELECT typeof(cash_balance) FROM accounts").fetchall() == [("text",)] * 6
        assert connection.execute("SELECT DISTINCT typeof(quantity), typeof(avg_cost) FROM holdings").fetchall() == [("text", "text")]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO holdings VALUES ('missing', 'SYNTH-X', '1', '1', '2026-01-01')")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO accounts VALUES ('bad', 'missing', 'cash', 'active', '0')")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO transactions VALUES ('bad', 'missing', 'SYNTH-X', 'buy', '1', '1', '2026-01-01')")
    finally:
        connection.close()


def test_profile_nulls_and_portfolio_decimal_precision(repository):
    profile = repository.get_profile("syn-user-005")
    assert profile.age_band is profile.risk_level is profile.region is None
    portfolio = repository.get_portfolio("syn-user-001")
    assert len(portfolio.accounts) == 2
    assert portfolio.accounts[0].cash_balance == Decimal("10000.10")
    holding = portfolio.accounts[0].holdings[0]
    assert holding.quantity == Decimal("10.125")
    assert holding.avg_cost == Decimal("12.34567890")
    data = portfolio.model_dump(mode="json")
    assert data["accounts"][0]["holdings"][0]["avg_cost"] == "12.34567890"
    assert portfolio.currency == "CNY"


def test_read_only_connection_and_missing_file(repository, tmp_path):
    with pytest.raises(RepositoryUnavailable):
        with repository._connect() as connection:
            connection.execute("DELETE FROM users")
    assert repository.get_profile("syn-user-001").user_id == "syn-user-001"
    path = tmp_path / "missing.db"
    with pytest.raises(RepositoryUnavailable):
        SQLiteUserDataRepository(path).get_profile("syn-user-001")
    assert not path.exists()


def test_sql_injection_is_only_a_bound_value(repository):
    injected = "syn-user-001' OR 1=1; DROP TABLE users; --"
    with pytest.raises(UserNotFound):
        repository.get_profile(injected)
    with pytest.raises(UserNotFound):
        repository.get_portfolio(injected)
    with pytest.raises(UserNotFound):
        repository.get_transactions(TransactionsInput(user_id=injected))
    assert repository.get_profile("syn-user-001").name_alias == "Synthetic Alpha"


def test_transactions_pagination_and_order(repository):
    first = repository.get_transactions(TransactionsInput(user_id="syn-user-001", limit=1))
    second = repository.get_transactions(TransactionsInput(user_id="syn-user-001", limit=1, offset=1))
    assert first.total == second.total == 4
    assert first.transactions[0].transaction_id == "syn-tx-001"
    assert second.transactions[0].transaction_id == "syn-tx-002"
    assert first.transactions[0].trade_time == second.transactions[0].trade_time
    assert first.model_dump(mode="json")["transactions"][0]["price"] == "12.34567890"
    page = repository.get_transactions(TransactionsInput(user_id="syn-user-001", offset=100))
    assert page.transactions == [] and page.total == 4


@pytest.mark.parametrize("start,end,expected", [
    ("2026-01-02T09:00:00Z", "2026-01-03T09:00:00Z", ["syn-tx-001", "syn-tx-002"]),
    ("2026-01-02T17:00:00+08:00", "2026-01-03T17:00:00+08:00", ["syn-tx-001", "syn-tx-002"]),
    ("2026-01-02T09:00:00.000001Z", "2026-01-04T09:00:00Z", ["syn-tx-003"]),
    (None, "2026-01-02T09:00:00Z", []),
    ("2026-01-04T09:00:00Z", None, ["syn-tx-004"]),
])
def test_transaction_time_boundaries(repository, start, end, expected):
    page = repository.get_transactions(TransactionsInput(user_id="syn-user-001", start_time=start, end_time=end))
    assert [row.transaction_id for row in page.transactions] == expected
    assert page.total == len(expected)


def test_empty_users_and_cross_user_isolation(repository):
    assert repository.get_portfolio("syn-user-002").accounts == []
    assert repository.get_portfolio("syn-user-003").accounts[0].holdings == []
    assert repository.get_transactions(TransactionsInput(user_id="syn-user-004")).total == 0
    page = repository.get_transactions(TransactionsInput(user_id="syn-user-006"))
    assert [row.transaction_id for row in page.transactions] == ["syn-tx-005"]
