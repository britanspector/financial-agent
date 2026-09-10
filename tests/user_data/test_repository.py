from decimal import Decimal
import sqlite3

import pytest

from financial_agent.user_data.models import MarginAccountInput
from financial_agent.user_data.repository import RepositoryUnavailable, SQLiteUserDataRepository, UserNotFound


def test_foreign_keys_and_business_tables(repository):
    connection = sqlite3.connect(repository.path)
    try:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 20
        assert connection.execute("SELECT COUNT(*) FROM margin_daily").fetchone()[0] == 1200
        assert connection.execute("SELECT typeof(period_net_pnl) FROM report_metrics LIMIT 1").fetchone()[0] == "text"
    finally:
        connection.close()


def test_customer_context_and_permissions(repository):
    context = repository.get_customer_context("syn-user-0001")
    assert context.user_id == "syn-user-0001"
    assert context.asset_bucket in {"lt_6m", "6m_8m", "8m_10m", "10m_12m", "ge_12m"}
    assert context.max_allowed_product_risk_level in {"R2", "R3", "R4", "R5"}


def test_margin_history_pagination_and_date_boundaries(repository):
    page = repository.get_margin_account(MarginAccountInput(user_id="syn-user-0001", start_date="2026-06-01", end_date="2026-06-30", limit=1))
    assert page.total == 29 and len(page.daily) == 1
    assert page.daily[0].trade_date == "2026-06-01"
    empty = repository.get_margin_account(MarginAccountInput(user_id="syn-user-0001", start_date="2027-01-01", limit=10))
    assert empty.total == 0 and empty.daily == []


def test_positions_and_analytics_have_coherent_children(repository):
    positions = repository.get_portfolio_positions("syn-user-0001")
    analytics = repository.get_portfolio_analytics("syn-user-0001")
    assert positions.stocks and positions.industries
    assert sum(item.weight for item in positions.stocks) == pytest.approx(1, abs=1e-6)
    assert analytics.report.period_net_pnl == Decimal(analytics.report.period_net_pnl)
    assert analytics.rank.peer_group_size >= 1


def test_read_only_connection_and_missing_file(repository, tmp_path):
    with pytest.raises(RepositoryUnavailable):
        with repository._connect() as connection:
            connection.execute("DELETE FROM users")
    path = tmp_path / "missing.db"
    with pytest.raises(RepositoryUnavailable):
        SQLiteUserDataRepository(path).get_customer_context("syn-user-0001")
    assert not path.exists()


def test_sql_injection_is_only_a_bound_value(repository):
    injected = "syn-user-0001' OR 1=1; DROP TABLE users; --"
    with pytest.raises(UserNotFound):
        repository.get_customer_context(injected)
    with pytest.raises(UserNotFound):
        repository.get_portfolio_positions(injected)
    assert repository.get_customer_context("syn-user-0001").user_id == "syn-user-0001"
