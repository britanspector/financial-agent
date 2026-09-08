"""Read-only, parameterized SQLite implementation of the repository contract."""

from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
import sqlite3
from typing import Protocol

from financial_agent.user_data.models import (
    Account, Holding, Portfolio, Profile, Transaction, TransactionPage, TransactionsInput,
)


class UserNotFound(Exception):
    """The requested synthetic user does not exist."""


class RepositoryUnavailable(Exception):
    """The local data store cannot be read."""


class UserDataRepository(Protocol):
    def get_profile(self, user_id: str) -> Profile: ...
    def get_portfolio(self, user_id: str) -> Portfolio: ...
    def get_transactions(self, query: TransactionsInput) -> TransactionPage: ...


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
    def _profile(connection, user_id: str) -> Profile:
        row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            raise UserNotFound()
        return Profile.model_validate(dict(row))

    def get_profile(self, user_id: str) -> Profile:
        with self._connect() as connection:
            return self._profile(connection, user_id)

    def get_portfolio(self, user_id: str) -> Portfolio:
        with self._connect() as connection:
            self._profile(connection, user_id)
            rows = connection.execute(
                "SELECT * FROM accounts WHERE user_id = ? ORDER BY account_id", (user_id,)
            ).fetchall()
            accounts = []
            for row in rows:
                holdings = connection.execute(
                    "SELECT symbol, quantity, avg_cost, updated_at FROM holdings "
                    "WHERE account_id = ? ORDER BY symbol", (row["account_id"],)
                ).fetchall()
                accounts.append(Account(**dict(row), holdings=[Holding(**dict(h)) for h in holdings]))
            return Portfolio(user_id=user_id, accounts=accounts)

    def get_transactions(self, query: TransactionsInput) -> TransactionPage:
        with self._connect() as connection:
            self._profile(connection, query.user_id)
            clauses = ["a.user_id = ?"]
            parameters = [query.user_id]
            for operator, value in ((">=", query.start_time), ("<", query.end_time)):
                if value is not None:
                    clauses.append(f"t.trade_time {operator} ?")
                    parameters.append(value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            selection = " FROM transactions t JOIN accounts a ON t.account_id = a.account_id WHERE "
            selection += " AND ".join(clauses)
            total = connection.execute("SELECT COUNT(*)" + selection, parameters).fetchone()[0]
            rows = connection.execute(
                "SELECT t.*" + selection + " ORDER BY t.trade_time, t.transaction_id LIMIT ? OFFSET ?",
                [*parameters, query.limit, query.offset],
            ).fetchall()
            return TransactionPage(
                user_id=query.user_id, transactions=[Transaction(**dict(row)) for row in rows],
                total=total, limit=query.limit, offset=query.offset,
            )
