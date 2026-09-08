"""Fixed synthetic records, deliberately small enough to inspect by hand."""

import os
from pathlib import Path
import sqlite3
import tempfile

STAMP = "2026-01-01T00:00:00.000000Z"

USERS = [
    ("syn-user-001", "Synthetic Alpha", "30-39", "medium", "synthetic-east", "standard", STAMP),
    ("syn-user-002", "Synthetic Beta", "20-29", "low", "synthetic-west", "standard", STAMP),
    ("syn-user-003", "Synthetic Gamma", "40-49", "low", "synthetic-north", "standard", STAMP),
    ("syn-user-004", "Synthetic Delta", "30-39", "high", "synthetic-south", "premium", STAMP),
    ("syn-user-005", "Synthetic Epsilon", None, None, None, "standard", STAMP),
    ("syn-user-006", "Synthetic Zeta", "50-59", "medium", "synthetic-east", "standard", STAMP),
]
ACCOUNTS = [
    ("syn-account-001", "syn-user-001", "cash", "active", "10000.10"),
    ("syn-account-002", "syn-user-001", "investment", "active", "2000.20"),
    ("syn-account-003", "syn-user-003", "investment", "active", "0.00"),
    ("syn-account-004", "syn-user-004", "investment", "active", "150.50"),
    ("syn-account-005", "syn-user-005", "cash", "active", "99.99"),
    ("syn-account-006", "syn-user-006", "investment", "closed", "0.00"),
]
HOLDINGS = [
    ("syn-account-001", "SYNTH-A", "10.125", "12.34567890", STAMP),
    ("syn-account-001", "SYNTH-B", "20", "50.00", STAMP),
    ("syn-account-002", "SYNTH-C", "3", "100.01", STAMP),
    ("syn-account-004", "SYNTH-A", "5", "11.20", STAMP),
]
TRANSACTIONS = [
    ("syn-tx-001", "syn-account-001", "SYNTH-A", "buy", "10.125", "12.34567890", "2026-01-02T09:00:00.000000Z"),
    ("syn-tx-002", "syn-account-002", "SYNTH-C", "buy", "3", "100.01", "2026-01-02T09:00:00.000000Z"),
    ("syn-tx-003", "syn-account-001", "SYNTH-B", "buy", "25", "49.50", "2026-01-03T09:00:00.000000Z"),
    ("syn-tx-004", "syn-account-001", "SYNTH-B", "sell", "5", "51.00", "2026-01-04T09:00:00.000000Z"),
    ("syn-tx-005", "syn-account-006", "SYNTH-D", "sell", "1", "88.00", "2026-01-05T09:00:00.000000Z"),
]

DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE users (
    user_id TEXT PRIMARY KEY NOT NULL, name_alias TEXT NOT NULL,
    age_band TEXT, risk_level TEXT, region TEXT,
    customer_tier TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    account_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'closed')),
    cash_balance TEXT NOT NULL
);
CREATE TABLE holdings (
    account_id TEXT NOT NULL REFERENCES accounts(account_id), symbol TEXT NOT NULL,
    quantity TEXT NOT NULL, avg_cost TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, symbol)
);
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY NOT NULL,
    account_id TEXT NOT NULL REFERENCES accounts(account_id), symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity TEXT NOT NULL, price TEXT NOT NULL, trade_time TEXT NOT NULL
);
CREATE INDEX accounts_user ON accounts(user_id);
CREATE INDEX transactions_account_time ON transactions(account_id, trade_time, transaction_id);
"""


def seed_user_data(path: Path, *, overwrite: bool = False) -> Path:
    """Publish a complete database atomically; never silently replace existing data."""
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Database exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".synthetic-", suffix=".db", dir=path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(DDL)
            with connection:
                connection.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", USERS)
                connection.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", ACCOUNTS)
                connection.executemany("INSERT INTO holdings VALUES (?, ?, ?, ?, ?)", HOLDINGS)
                connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)", TRANSACTIONS)
        finally:
            connection.close()
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link fails if another writer created the target in the meantime.
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
