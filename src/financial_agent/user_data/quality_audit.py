"""Read-only quality checks for generated synthetic margin-user databases."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import itertools
import math
import random
import sqlite3
from typing import Any


EXPECTED_TABLES = (
    "users", "accounts", "holdings", "transactions", "query_base_info", "margin_info",
    "report_metrics", "margin_daily", "industry_position", "stock_position", "factors", "user_return_rank",
)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return round(numerator / denominator, 4) if denominator else None


def _scalar(connection: sqlite3.Connection, statement: str, parameters: tuple = ()) -> int:
    return int(connection.execute(statement, parameters).fetchone()[0])


def audit_synthetic_data(path: Path, *, sample_size: int = 10, seed: int = 20260910) -> dict[str, Any]:
    """Return a JSON-serializable read-only audit report for a generated database."""
    with sqlite3.connect(Path(path).resolve()) as connection:
        connection.row_factory = sqlite3.Row
        table_counts = {name: _scalar(connection, f"SELECT COUNT(*) FROM {name}") for name in EXPECTED_TABLES}
        user_count = table_counts["users"]
        null_counts: dict[str, int] = {}
        for table in EXPECTED_TABLES:
            for column in connection.execute(f"PRAGMA table_info({table})"):
                column_name = column[1]
                null_count = _scalar(connection, f"SELECT COUNT(*) FROM {table} WHERE {column_name} IS NULL")
                if null_count:
                    null_counts[f"{table}.{column_name}"] = null_count

        abnormal = {
            "invalid_assets": _scalar(connection, "SELECT COUNT(*) FROM query_base_info WHERE CAST(total_asset AS REAL) <= 0 OR CAST(cash_asset AS REAL) < 0 OR CAST(investable_asset AS REAL) < CAST(cash_asset AS REAL)"),
            "invalid_margin": _scalar(connection, "SELECT COUNT(*) FROM margin_info WHERE CAST(credit_limit AS REAL) <= 0 OR CAST(financing_balance AS REAL) < 0 OR CAST(securities_lending_balance AS REAL) < 0 OR maintenance_margin_ratio <= 0 OR credit_utilization < 0 OR credit_utilization > 1"),
            "invalid_report_metrics": _scalar(connection, "SELECT COUNT(*) FROM report_metrics WHERE volatility < 0 OR max_drawdown > 0 OR win_rate < 0 OR win_rate > 1 OR turnover_rate < 0 OR average_holding_days <= 0"),
            "invalid_factor_values": _scalar(connection, "SELECT COUNT(*) FROM factors WHERE market_beta <= 0 OR leverage_exposure < 0 OR concentration_exposure < 0"),
            "invalid_rank_values": _scalar(connection, "SELECT COUNT(*) FROM user_return_rank WHERE return_rank < 1 OR return_rank > peer_group_size OR drawdown_rank < 1 OR drawdown_rank > peer_group_size OR risk_adjusted_rank < 1 OR risk_adjusted_rank > peer_group_size"),
            "non_60_day_histories": _scalar(connection, "SELECT COUNT(*) FROM (SELECT user_id, COUNT(*) AS days FROM margin_daily GROUP BY user_id HAVING days <> 60)"),
            "ledger_identity_mismatches": _scalar(connection, """
                SELECT COUNT(*) FROM report_metrics
                WHERE ABS(CAST(period_net_pnl AS REAL) - (
                    CAST(realized_pnl AS REAL) + CAST(ending_unrealized_pnl AS REAL)
                    - CAST(beginning_unrealized_pnl AS REAL) - CAST(financing_interest AS REAL)
                    - CAST(securities_lending_fee AS REAL) - CAST(transaction_fees AS REAL)
                    - CAST(other_fees AS REAL))) > 0.70
            """),
            "return_formula_mismatches": _scalar(connection, "SELECT COUNT(*) FROM report_metrics WHERE ABS(portfolio_return - CAST(period_net_pnl AS REAL) / CAST(beginning_equity AS REAL)) > 0.0000001"),
        }
        foreign_key_violations = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]

        snapshot_mismatch = _scalar(connection, """
            SELECT COUNT(*) FROM margin_info m JOIN margin_daily d
              ON m.user_id = d.user_id AND m.snapshot_date = d.trade_date
            WHERE ABS(CAST(m.collateral_market_value AS REAL) - CAST(d.collateral_market_value AS REAL)) > 0.01
               OR ABS(CAST(m.financing_balance AS REAL) - CAST(d.financing_balance AS REAL)) > 0.01
               OR ABS(CAST(m.securities_lending_balance AS REAL) - CAST(d.securities_lending_balance AS REAL)) > 0.01
               OR ABS(m.maintenance_margin_ratio - d.maintenance_margin_ratio) > 0.0000001
               OR ABS(m.credit_utilization - d.credit_utilization) > 0.0000001
               OR m.margin_call_flag <> d.margin_call_flag
               OR m.forced_liquidation_flag <> d.forced_liquidation_flag
        """)
        stock_industry_mismatch = _scalar(connection, """
            WITH stock AS (
              SELECT user_id, industry_code, SUM(CAST(market_value AS REAL)) AS market_value,
                     SUM(CAST(cost_value AS REAL)) AS cost_value, SUM(weight) AS weight
              FROM stock_position GROUP BY user_id, industry_code
            )
            SELECT COUNT(*) FROM stock s JOIN industry_position i
              ON s.user_id = i.user_id AND s.industry_code = i.industry_code
            WHERE ABS(s.market_value - CAST(i.market_value AS REAL)) > 0.11
               OR ABS(s.cost_value - CAST(i.cost_value AS REAL)) > 0.11
               OR ABS(s.weight - i.weight) > 0.000001
        """)
        missing_industry_rows = _scalar(connection, """
            SELECT COUNT(*) FROM (
              SELECT user_id, industry_code FROM stock_position GROUP BY user_id, industry_code
              EXCEPT SELECT user_id, industry_code FROM industry_position
            )
        """)
        weight_mismatch = _scalar(connection, """
            SELECT COUNT(*) FROM (
              SELECT user_id, SUM(weight) AS total_weight FROM stock_position GROUP BY user_id HAVING ABS(total_weight - 1) > 0.000001
              UNION ALL
              SELECT user_id, SUM(weight) AS total_weight FROM industry_position GROUP BY user_id HAVING ABS(total_weight - 1) > 0.000001
            )
        """)
        daily_pnl_mismatch = _scalar(connection, """
            SELECT COUNT(*) FROM report_metrics r JOIN (
              SELECT user_id, SUM(CAST(daily_profit_loss AS REAL)) AS pnl FROM margin_daily GROUP BY user_id
            ) d ON r.user_id = d.user_id
            WHERE ABS(CAST(r.period_net_pnl AS REAL) - d.pnl) > 0.70
        """)
        daily_formula_mismatch = _scalar(connection, """
            SELECT COUNT(*) FROM margin_daily
            WHERE ABS(CAST(daily_profit_loss AS REAL) - (
                CAST(daily_realized_pnl AS REAL) + CAST(daily_unrealized_pnl_change AS REAL)
                - CAST(daily_total_costs AS REAL))) > 0.02
        """)
        rank_group_mismatch = _scalar(connection, """
            SELECT COUNT(*) FROM user_return_rank r JOIN (
                SELECT region || '|' || asset_bucket AS peer_group_key, COUNT(*) AS n
                FROM users JOIN query_base_info USING(user_id) GROUP BY peer_group_key
            ) g USING(peer_group_key)
            WHERE r.peer_group_size <> g.n
        """)

        rows = connection.execute("""
            SELECT r.user_id, r.portfolio_return, r.volatility, r.max_drawdown, r.turnover_rate,
                   r.average_holding_days, r.sharpe_ratio, m.credit_utilization,
                   f.leverage_exposure, f.concentration_exposure, f.volatility_exposure,
                   k.peer_group_key, k.peer_group_size, k.return_rank, k.drawdown_rank, k.risk_adjusted_rank,
                   s.stock_count, s.max_stock_weight
            FROM report_metrics r
            JOIN margin_info m USING(user_id)
            JOIN factors f USING(user_id)
            JOIN user_return_rank k USING(user_id)
            JOIN (
              SELECT user_id, COUNT(*) AS stock_count, MAX(weight) AS max_stock_weight
              FROM stock_position GROUP BY user_id
            ) s USING(user_id)
            ORDER BY r.user_id
        """).fetchall()
        returns = [row["portfolio_return"] for row in rows]
        drawdowns = [row["max_drawdown"] for row in rows]
        risk_adjusted = [row["portfolio_return"] / max(0.001, abs(row["max_drawdown"])) for row in rows]
        rank_mismatches = {"return": 0, "drawdown": 0, "risk_adjusted": 0}
        for peer_group, peer_rows in itertools.groupby(sorted(rows, key=lambda row: row["peer_group_key"]), key=lambda row: row["peer_group_key"]):
            group_rows = list(peer_rows)
            for field, rank_field in (("portfolio_return", "return_rank"), ("max_drawdown", "drawdown_rank"), ("sharpe_ratio", "risk_adjusted_rank")):
                expected = {}
                previous = object()
                current_rank = 0
                for position, row in enumerate(sorted(group_rows, key=lambda item: item[field], reverse=True), 1):
                    if row[field] != previous:
                        current_rank = position
                        previous = row[field]
                    expected[row["user_id"]] = current_rank
                label = {"return_rank": "return", "drawdown_rank": "drawdown", "risk_adjusted_rank": "risk_adjusted"}[rank_field]
                rank_mismatches[label] += sum(row[rank_field] != expected[row["user_id"]] for row in group_rows)
        ranked_by_return = sorted(rows, key=lambda row: row["return_rank"])
        rank_check = {
            "peer_group_rank_mismatches": rank_mismatches,
            "return_rank_correlation": _correlation(returns, [-row["return_rank"] for row in rows]),
            "drawdown_rank_correlation": _correlation(drawdowns, [-row["drawdown_rank"] for row in rows]),
            "risk_adjusted_rank_correlation": _correlation(risk_adjusted, [-row["risk_adjusted_rank"] for row in rows]),
        }
        factor_relationships = {
            "leverage_exposure_vs_credit_utilization": _correlation([row["leverage_exposure"] for row in rows], [row["credit_utilization"] for row in rows]),
            "concentration_exposure_vs_max_stock_weight": _correlation([row["concentration_exposure"] for row in rows], [row["max_stock_weight"] for row in rows]),
            "volatility_exposure_vs_report_volatility": _correlation([row["volatility_exposure"] for row in rows], [row["volatility"] for row in rows]),
        }
        position_rows = connection.execute("""
            SELECT r.portfolio_return, r.max_drawdown, m.credit_utilization,
                   (SUM(CAST(s.market_value AS REAL)) - SUM(CAST(s.cost_value AS REAL)))
                     / SUM(CAST(s.cost_value AS REAL)) AS position_return
            FROM report_metrics r JOIN margin_info m USING(user_id)
            JOIN stock_position s USING(user_id)
            GROUP BY r.user_id
        """).fetchall()
        report_relationships = {
            "report_return_vs_position_profit_pct": _correlation(
                [row["portfolio_return"] for row in position_rows], [row["position_return"] for row in position_rows],
            ),
            "credit_utilization_vs_absolute_drawdown": _correlation(
                [row["credit_utilization"] for row in position_rows], [-row["max_drawdown"] for row in position_rows],
            ),
        }
        diversity = {
            "unique_asset_values": _scalar(connection, "SELECT COUNT(DISTINCT total_asset) FROM query_base_info"),
            "unique_metric_vectors": _scalar(connection, "SELECT COUNT(DISTINCT printf('%.6f|%.6f|%.6f|%.6f', portfolio_return, max_drawdown, turnover_rate, average_holding_days)) FROM report_metrics"),
            "stock_count_distribution": dict(Counter(row["stock_count"] for row in rows)),
            "max_stock_weight_min_max": [round(min(row["max_stock_weight"] for row in rows), 4), round(max(row["max_stock_weight"] for row in rows), 4)],
            "credit_utilization_min_max": [round(min(row["credit_utilization"] for row in rows), 4), round(max(row["credit_utilization"] for row in rows), 4)],
            "return_min_max": [round(min(returns), 4), round(max(returns), 4)],
            "drawdown_min_max": [round(min(drawdowns), 4), round(max(drawdowns), 4)],
        }

        label_columns = []
        for table in EXPECTED_TABLES:
            for column in connection.execute(f"PRAGMA table_info({table})"):
                column_name = column[1].lower()
                if any(token in column_name for token in ("archetype", "persona", "personality", "user_type")):
                    label_columns.append(f"{table}.{column[1]}")

        selected_ids = sorted(row["user_id"] for row in rows)
        sample_rng = random.Random(seed ^ 0xC0DE)
        sample_ids = sorted(sample_rng.sample(selected_ids, min(sample_size, len(selected_ids))))
        samples = [dict(row) for row in connection.execute("""
            SELECT q.user_id, q.total_asset, q.cash_asset, q.risk_tolerance_score,
                   m.credit_utilization, m.maintenance_margin_ratio,
                   r.portfolio_return, r.volatility, r.max_drawdown, r.turnover_rate,
                   r.average_holding_days, f.value_exposure, f.momentum_exposure,
                   f.leverage_exposure, f.concentration_exposure, k.return_rank,
                   k.drawdown_rank, k.risk_adjusted_rank,
                   s.stock_count, s.max_stock_weight, i.max_industry_weight
            FROM query_base_info q JOIN margin_info m USING(user_id)
            JOIN report_metrics r USING(user_id) JOIN factors f USING(user_id)
            JOIN user_return_rank k USING(user_id)
            JOIN (SELECT user_id, COUNT(*) AS stock_count, MAX(weight) AS max_stock_weight FROM stock_position GROUP BY user_id) s USING(user_id)
            JOIN (SELECT user_id, MAX(weight) AS max_industry_weight FROM industry_position GROUP BY user_id) i USING(user_id)
            WHERE q.user_id IN ({}) ORDER BY q.user_id
        """.format(",".join("?" for _ in sample_ids)), sample_ids)]

    return {
        "path": str(Path(path).resolve()), "table_counts": table_counts,
        "null_counts": null_counts, "abnormal_counts": abnormal,
        "foreign_key_violations": foreign_key_violations,
        "consistency": {
            "margin_info_vs_snapshot_daily_mismatches": snapshot_mismatch,
            "stock_vs_industry_mismatches": stock_industry_mismatch,
            "stock_industries_missing_rollup": missing_industry_rows,
            "position_weight_mismatches": weight_mismatch,
            "report_vs_sum_daily_pnl_mismatches": daily_pnl_mismatch,
            "daily_pnl_formula_mismatches": daily_formula_mismatch,
            "rank_peer_group_size_mismatches": rank_group_mismatch,
        },
        "diversity": diversity, "factor_relationships": factor_relationships,
        "report_relationships": report_relationships,
        "rank_check": rank_check, "generation_label_columns": label_columns,
        "samples": samples,
    }
