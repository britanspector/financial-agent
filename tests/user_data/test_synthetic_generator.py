import sqlite3

import pytest

from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.quality_audit import audit_synthetic_data
from financial_agent.user_data.synthetic_generator import generate_synthetic_data


def test_twenty_user_generator_is_repeatable_and_keeps_archetypes_internal(tmp_path):
    first = generate_synthetic_data(tmp_path / "first.db", seed=20260910, user_count=20)
    second = generate_synthetic_data(tmp_path / "second.db", seed=20260910, user_count=20)
    assert first.summaries == second.summaries
    assert len(first.summaries) == 20

    with sqlite3.connect(first.path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"query_base_info", "margin_info", "report_metrics", "margin_daily",
                "industry_position", "stock_position", "factors", "user_return_rank"} <= tables
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 20
        assert connection.execute("SELECT COUNT(*) FROM margin_daily").fetchone()[0] == 20 * 60
        columns = [row[1].lower() for row in connection.execute("PRAGMA table_info(query_base_info)")]
        assert not any(token in column for column in columns for token in ("archetype", "persona", "user_type"))


def test_generated_tables_obey_cross_table_constraints_and_old_tools_read_them(tmp_path):
    result = generate_synthetic_data(tmp_path / "generated.db", seed=20260910, user_count=20)
    with sqlite3.connect(result.path) as connection:
        bad_stock = connection.execute(
            "SELECT COUNT(*) FROM stock_position WHERE ABS(market_value - quantity * market_price) > 0.05"
        ).fetchone()[0]
        assert bad_stock == 0
        bad_margin = connection.execute(
            "SELECT COUNT(*) FROM margin_info WHERE ABS(credit_utilization - "
            "(CAST(financing_balance AS REAL) + CAST(securities_lending_balance AS REAL)) / CAST(credit_limit AS REAL)) > 0.0001"
        ).fetchone()[0]
        assert bad_margin == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM user_return_rank WHERE return_rank < 1 OR return_rank > peer_group_size"
        ).fetchone()[0] == 0

    repository = SQLiteUserDataRepository(result.path)
    assert repository.get_customer_context("syn-user-0003").user_id == "syn-user-0003"
    assert repository.get_portfolio_positions("syn-user-0003").stocks
    assert repository.get_portfolio_analytics("syn-user-0003").rank.peer_group_size >= 1


def test_quality_audit_is_read_only_and_reports_no_structural_failures(tmp_path):
    result = generate_synthetic_data(tmp_path / "generated.db", seed=20260910, user_count=20)
    audit = audit_synthetic_data(result.path)
    assert audit["table_counts"]["users"] == 20
    assert audit["null_counts"] == {}
    assert all(value == 0 for value in audit["abnormal_counts"].values())
    assert audit["foreign_key_violations"] == []
    assert all(value == 0 for value in audit["consistency"].values())
    assert audit["generation_label_columns"] == []


@pytest.mark.integration
def test_full_2000_user_generation_reproducibility_and_quality(tmp_path):
    first = generate_synthetic_data(tmp_path / "first.db", seed=20260910)
    second = generate_synthetic_data(tmp_path / "second.db", seed=20260910)

    def logical_dump(path):
        with sqlite3.connect(path) as connection:
            return "\n".join(connection.iterdump())

    first_dump = logical_dump(first.path)
    assert first_dump == logical_dump(second.path)
    assert len(first.summaries) == 2000
    assert first.primary_distribution == {
        "diversified_steady": 360, "long_term_value": 280, "high_leverage": 220,
        "single_stock_concentrated": 180, "industry_concentrated": 200,
        "short_term_turnover": 240, "high_return_drawdown": 200, "cash_defensive": 320,
    }
    audit = audit_synthetic_data(first.path)
    assert audit["table_counts"]["users"] == 2000
    assert audit["null_counts"] == {}
    assert all(value == 0 for value in audit["abnormal_counts"].values())
    assert audit["foreign_key_violations"] == []
    assert all(value == 0 for value in audit["consistency"].values())
    assert audit["generation_label_columns"] == []
