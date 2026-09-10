"""Read-only, parameterized SQLite implementation for business Tool views."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Protocol

from financial_agent.user_data.models import (
    CustomerContext, Factors, IndustryPosition, MarginAccount, MarginAccountInput, MarginDaily,
    MarginInfo, PortfolioAnalytics, PortfolioPositions, ReportMetrics, StockPosition,
    UserReturnRank,
)


class UserNotFound(Exception):
    """The requested synthetic user does not exist."""


class RepositoryUnavailable(Exception):
    """The local data store cannot be read."""


class UserDataRepository(Protocol):
    def get_customer_context(self, user_id: str) -> CustomerContext: ...
    def get_margin_account(self, query: MarginAccountInput) -> MarginAccount: ...
    def get_portfolio_positions(self, user_id: str) -> PortfolioPositions: ...
    def get_portfolio_analytics(self, user_id: str) -> PortfolioAnalytics: ...


class SQLiteUserDataRepository:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()

    @contextmanager
    def _connect(self):
        try:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=0)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("BEGIN")
                yield connection
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise RepositoryUnavailable("Synthetic data store unavailable") from exc

    @staticmethod
    def _require_user(connection: sqlite3.Connection, user_id: str) -> None:
        if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
            raise UserNotFound()

    def get_customer_context(self, user_id: str) -> CustomerContext:
        with self._connect() as connection:
            row = connection.execute("""
                SELECT u.user_id, u.name_alias, u.age_band, u.risk_level, u.region,
                       u.customer_tier, u.created_at, q.snapshot_date, q.asset_bucket,
                       q.investable_asset, q.total_asset, q.cash_asset,
                       q.risk_tolerance_score, q.investment_experience_years,
                       q.investment_horizon, q.liquidity_need_score,
                       q.active_trading_days_90d, q.trade_enabled, q.margin_enabled,
                       q.short_selling_enabled, q.max_allowed_product_risk_level
                FROM users u JOIN query_base_info q USING(user_id)
                WHERE u.user_id = ?
            """, (user_id,)).fetchone()
            if row is None:
                raise UserNotFound()
            return CustomerContext.model_validate(dict(row))

    def get_margin_account(self, query: MarginAccountInput) -> MarginAccount:
        with self._connect() as connection:
            self._require_user(connection, query.user_id)
            info_row = connection.execute("SELECT * FROM margin_info WHERE user_id = ?", (query.user_id,)).fetchone()
            if info_row is None:
                raise RepositoryUnavailable("Synthetic margin account unavailable")
            clauses = ["user_id = ?"]
            params: list[object] = [query.user_id]
            if query.start_date is not None:
                clauses.append("trade_date >= ?")
                params.append(query.start_date.isoformat())
            if query.end_date is not None:
                clauses.append("trade_date < ?")
                params.append(query.end_date.isoformat())
            where = " AND ".join(clauses)
            total = connection.execute(f"SELECT COUNT(*) FROM margin_daily WHERE {where}", params).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM margin_daily WHERE {where} ORDER BY trade_date LIMIT ? OFFSET ?",
                [*params, query.limit, query.offset],
            ).fetchall()
            return MarginAccount(
                user_id=query.user_id,
                info=MarginInfo.model_validate(dict(info_row)),
                daily=[MarginDaily.model_validate(dict(row)) for row in rows],
                total=total,
                limit=query.limit,
                offset=query.offset,
                start_date=query.start_date.isoformat() if query.start_date else None,
                end_date=query.end_date.isoformat() if query.end_date else None,
            )

    def get_portfolio_positions(self, user_id: str) -> PortfolioPositions:
        with self._connect() as connection:
            self._require_user(connection, user_id)
            rows = connection.execute(
                "SELECT * FROM industry_position WHERE user_id = ? ORDER BY position_rank, industry_code", (user_id,)
            ).fetchall()
            stocks = connection.execute(
                "SELECT * FROM stock_position WHERE user_id = ? ORDER BY weight DESC, stock_code", (user_id,)
            ).fetchall()
            snapshot = (rows[0]["snapshot_date"] if rows else stocks[0]["snapshot_date"] if stocks else None)
            return PortfolioPositions(
                user_id=user_id,
                snapshot_date=snapshot or "",
                industries=[IndustryPosition.model_validate(dict(row)) for row in rows],
                stocks=[StockPosition.model_validate(dict(row)) for row in stocks],
            )

    def get_portfolio_analytics(self, user_id: str) -> PortfolioAnalytics:
        with self._connect() as connection:
            self._require_user(connection, user_id)
            report = connection.execute("SELECT * FROM report_metrics WHERE user_id = ?", (user_id,)).fetchone()
            factors = connection.execute("SELECT * FROM factors WHERE user_id = ?", (user_id,)).fetchone()
            rank = connection.execute("SELECT * FROM user_return_rank WHERE user_id = ?", (user_id,)).fetchone()
            if report is None or factors is None or rank is None:
                raise RepositoryUnavailable("Synthetic analytics unavailable")
            return PortfolioAnalytics(
                user_id=user_id,
                report=ReportMetrics.model_validate(dict(report)),
                factors=Factors.model_validate(dict(factors)),
                rank=UserReturnRank.model_validate(dict(rank)),
            )
