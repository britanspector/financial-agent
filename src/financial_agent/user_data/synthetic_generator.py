"""Deterministic, inspectable synthetic margin-account data generator.

Archetypes in this module are generation-only rules.  They are intentionally
not persisted in any table and are not part of the user-data Tool contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import os
import random
import sqlite3
import tempfile
from typing import Any


DEFAULT_SEED = 20260910
USER_COUNT = 2_000
SNAPSHOT = date(2026, 6, 30)
TRADING_DAYS = 60

INDUSTRIES = ("technology", "healthcare", "consumer", "finance", "energy", "utilities", "manufacturing", "telecom")
SYMBOLS = tuple(f"SYN{i:03d}" for i in range(1, 81))


@dataclass(frozen=True)
class GenerationResult:
    path: Path
    seed: int
    summaries: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    constraints: list[str]
    primary_distribution: dict[str, int]


@dataclass(frozen=True)
class LatentProfile:
    # Generation-only values.  Never serialize this object to SQLite.
    primary: str
    momentum: bool
    value_quality: bool
    growth: bool
    event_driven: bool
    contrarian: bool


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
    account_type TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('active', 'closed')),
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

CREATE TABLE query_base_info (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id), snapshot_date TEXT NOT NULL,
    investable_asset TEXT NOT NULL, total_asset TEXT NOT NULL, cash_asset TEXT NOT NULL,
    risk_tolerance_score REAL NOT NULL, investment_experience_years INTEGER NOT NULL,
    investment_horizon TEXT NOT NULL, liquidity_need_score REAL NOT NULL,
    active_trading_days_90d INTEGER NOT NULL, trade_enabled INTEGER NOT NULL,
    margin_enabled INTEGER NOT NULL, short_selling_enabled INTEGER NOT NULL,
    max_allowed_product_risk_level TEXT NOT NULL, asset_bucket TEXT NOT NULL
);
CREATE TABLE margin_info (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id), snapshot_date TEXT NOT NULL,
    margin_account_id TEXT NOT NULL, credit_limit TEXT NOT NULL,
    financing_balance TEXT NOT NULL, securities_lending_balance TEXT NOT NULL,
    collateral_market_value TEXT NOT NULL, cash_balance TEXT NOT NULL,
    available_credit TEXT NOT NULL, maintenance_margin_ratio REAL NOT NULL,
    warning_line REAL NOT NULL, liquidation_line REAL NOT NULL,
    credit_utilization REAL NOT NULL, margin_call_flag INTEGER NOT NULL,
    forced_liquidation_flag INTEGER NOT NULL
);
CREATE TABLE report_metrics (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id), report_date TEXT NOT NULL,
    period_start TEXT NOT NULL, period_end TEXT NOT NULL,
    beginning_equity TEXT NOT NULL, period_net_pnl TEXT NOT NULL,
    realized_pnl TEXT NOT NULL, beginning_unrealized_pnl TEXT NOT NULL,
    ending_unrealized_pnl TEXT NOT NULL, financing_interest TEXT NOT NULL,
    securities_lending_fee TEXT NOT NULL, transaction_fees TEXT NOT NULL,
    other_fees TEXT NOT NULL, portfolio_return REAL NOT NULL, annualized_return REAL NOT NULL,
    volatility REAL NOT NULL,
    max_drawdown REAL NOT NULL, sharpe_ratio REAL NOT NULL,
    win_rate REAL NOT NULL, turnover_rate REAL NOT NULL,
    average_holding_days REAL NOT NULL, profit_trade_ratio REAL NOT NULL,
    margin_contribution REAL NOT NULL, benchmark_excess_return REAL NOT NULL
);
CREATE TABLE margin_daily (
    user_id TEXT NOT NULL REFERENCES users(user_id), trade_date TEXT NOT NULL,
    collateral_market_value TEXT NOT NULL, financing_balance TEXT NOT NULL,
    securities_lending_balance TEXT NOT NULL, cash_balance TEXT NOT NULL,
    maintenance_margin_ratio REAL NOT NULL, credit_utilization REAL NOT NULL,
    daily_realized_pnl TEXT NOT NULL, daily_unrealized_pnl_change TEXT NOT NULL,
    daily_total_costs TEXT NOT NULL, daily_profit_loss TEXT NOT NULL, daily_return REAL NOT NULL,
    margin_call_flag INTEGER NOT NULL, forced_liquidation_flag INTEGER NOT NULL,
    collateral_injection_amount TEXT NOT NULL, deleveraging_amount TEXT NOT NULL,
    PRIMARY KEY (user_id, trade_date)
);
CREATE TABLE industry_position (
    user_id TEXT NOT NULL REFERENCES users(user_id), snapshot_date TEXT NOT NULL,
    industry_code TEXT NOT NULL, market_value TEXT NOT NULL, weight REAL NOT NULL,
    cost_value TEXT NOT NULL, profit_loss TEXT NOT NULL, profit_loss_pct REAL NOT NULL,
    position_rank INTEGER NOT NULL, PRIMARY KEY (user_id, industry_code)
);
CREATE TABLE stock_position (
    user_id TEXT NOT NULL REFERENCES users(user_id), snapshot_date TEXT NOT NULL,
    stock_code TEXT NOT NULL, industry_code TEXT NOT NULL, quantity TEXT NOT NULL,
    market_price TEXT NOT NULL, market_value TEXT NOT NULL, cost_price TEXT NOT NULL,
    cost_value TEXT NOT NULL, weight REAL NOT NULL, profit_loss TEXT NOT NULL,
    profit_loss_pct REAL NOT NULL, holding_days INTEGER NOT NULL,
    last_trade_date TEXT NOT NULL, PRIMARY KEY (user_id, stock_code)
);
CREATE TABLE factors (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id), snapshot_date TEXT NOT NULL,
    market_beta REAL NOT NULL, size_exposure REAL NOT NULL, value_exposure REAL NOT NULL,
    growth_exposure REAL NOT NULL, momentum_exposure REAL NOT NULL,
    quality_exposure REAL NOT NULL, volatility_exposure REAL NOT NULL,
    liquidity_exposure REAL NOT NULL, leverage_exposure REAL NOT NULL,
    concentration_exposure REAL NOT NULL
);
CREATE TABLE user_return_rank (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id), rank_date TEXT NOT NULL,
    peer_group_key TEXT NOT NULL, peer_group_size INTEGER NOT NULL,
    period_return REAL NOT NULL, benchmark_excess_return REAL NOT NULL,
    return_rank INTEGER NOT NULL, return_percentile REAL NOT NULL,
    drawdown_rank INTEGER NOT NULL, risk_adjusted_rank INTEGER NOT NULL
);
"""


PRIMARY_WEIGHTS = {
    "diversified_steady": 0.18,
    "long_term_value": 0.14,
    "high_leverage": 0.11,
    "single_stock_concentrated": 0.09,
    "industry_concentrated": 0.10,
    "short_term_turnover": 0.12,
    "high_return_drawdown": 0.10,
    "cash_defensive": 0.16,
}


def _money(value: float) -> str:
    return f"{max(0.0, value):.2f}"


def _signed_money(value: float) -> str:
    return f"{value:.2f}"


def _price(value: float) -> str:
    return f"{max(0.01, value):.2f}"


def _date_text(value: date) -> str:
    return value.isoformat()


def _asset_bucket(asset: float) -> str:
    if asset < 6_000_000:
        return "lt_6m"
    if asset < 8_000_000:
        return "6m_8m"
    if asset < 10_000_000:
        return "8m_10m"
    if asset < 12_000_000:
        return "10m_12m"
    return "ge_12m"


def _allocate_primaries(user_count: int, seed: int) -> list[str]:
    """Allocate exact-near target primary proportions, then deterministically shuffle."""
    raw = {name: weight * user_count for name, weight in PRIMARY_WEIGHTS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = user_count - sum(counts.values())
    for name in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)[:remainder]:
        counts[name] += 1
    assigned = [name for name, count in counts.items() for _ in range(count)]
    random.Random(seed ^ 0xA5A5).shuffle(assigned)
    return assigned


def _latent(primary: str, rng: random.Random) -> LatentProfile:
    # These values are generation-only and deliberately not written to SQLite.
    return LatentProfile(
        primary=primary,
        momentum=rng.random() < 0.18,
        value_quality=rng.random() < 0.16,
        growth=rng.random() < 0.12,
        event_driven=rng.random() < 0.08,
        contrarian=rng.random() < 0.10,
    )


def _parameters(latent: LatentProfile) -> dict[str, float | int | str]:
    primary = latent.primary
    base: dict[str, float | int | str] = {
        "asset": 8_000_000.0,
        "cash_ratio": 0.20,
        "leverage": 0.10,
        "concentration": 0.18,
        "stock_count": 14,
        "holding_days": 90,
        "risk": 0.50,
        "return_bias": 0.004,
        "vol": 0.018,
    }
    overrides = {
        "diversified_steady": (0.24, 0.08, 0.10, 22, 120, 0.38, 0.003, 0.010),
        "long_term_value": (0.18, 0.12, 0.16, 12, 260, 0.46, 0.006, 0.013),
        "high_leverage": (0.10, 0.72, 0.28, 10, 45, 0.78, 0.010, 0.030),
        "single_stock_concentrated": (0.12, 0.22, 0.68, 5, 80, 0.70, 0.009, 0.026),
        "industry_concentrated": (0.14, 0.18, 0.42, 10, 75, 0.66, 0.008, 0.022),
        "short_term_turnover": (0.12, 0.25, 0.22, 16, 8, 0.64, 0.006, 0.025),
        "high_return_drawdown": (0.08, 0.46, 0.38, 8, 35, 0.82, 0.014, 0.040),
        "cash_defensive": (0.46, 0.03, 0.10, 8, 150, 0.28, 0.002, 0.008),
    }
    cash, leverage, concentration, count, holding, risk, ret, vol = overrides[primary]
    base.update(cash_ratio=cash, leverage=leverage, concentration=concentration,
                stock_count=count, holding_days=holding, risk=risk,
                return_bias=ret, vol=vol)
    if latent.momentum:
        base["return_bias"] = float(base["return_bias"]) + 0.002
        base["vol"] = float(base["vol"]) + 0.004
        base["holding_days"] = max(3, int(base["holding_days"]) // 2)
    if latent.value_quality:
        base["return_bias"] = float(base["return_bias"]) + 0.001
        base["holding_days"] = int(base["holding_days"]) + 35
    if latent.growth:
        base["vol"] = float(base["vol"]) + 0.006
        base["concentration"] = min(0.85, float(base["concentration"]) + 0.08)
    if latent.contrarian:
        base["return_bias"] = float(base["return_bias"]) + 0.001
    return base


def _build_rows(rng: random.Random, index: int, latent: LatentProfile) -> tuple[list[tuple], dict[str, Any]]:
    p = _parameters(latent)
    user_id = f"syn-user-{index + 1:04d}"
    account_id = f"syn-margin-{index + 1:04d}"
    asset = float(p["asset"]) * rng.uniform(0.70, 1.45)
    cash = asset * float(p["cash_ratio"]) * rng.uniform(0.85, 1.15)
    investable = asset - cash * 0.25
    risk_score = min(0.99, max(0.05, float(p["risk"]) + rng.uniform(-0.06, 0.06)))
    industries = list(INDUSTRIES)
    rng.shuffle(industries)
    count = int(p["stock_count"])
    if latent.primary == "single_stock_concentrated":
        weights = [float(p["concentration"])] + [rng.uniform(0.02, 0.08) for _ in range(count - 1)]
    elif latent.primary == "industry_concentrated":
        weights = [rng.uniform(0.04, 0.12) for _ in range(count)]
        weights[0] += float(p["concentration"])
    else:
        weights = [rng.uniform(0.6, 1.4) for _ in range(count)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    if latent.primary == "single_stock_concentrated":
        weights[0] = float(p["concentration"])
        rest = 1.0 - weights[0]
        tail = sum(weights[1:])
        weights[1:] = [w / tail * rest for w in weights[1:]]
    elif latent.primary == "industry_concentrated":
        weights[0] = min(0.75, 0.42 + rng.uniform(0.0, 0.12))
        rest = 1.0 - weights[0]
        tail = sum(weights[1:])
        weights[1:] = [w / tail * rest for w in weights[1:]]

    stock_rows: list[tuple] = []
    industry_totals: dict[str, list[float]] = {}
    positions: list[dict[str, Any]] = []
    for stock_no, weight in enumerate(weights):
        industry = industries[0] if latent.primary == "industry_concentrated" and stock_no < max(2, count // 2) else industries[stock_no % len(industries)]
        price = round(rng.uniform(25, 420), 2)
        market_value = investable * 0.78 * weight
        quantity = round(max(1.0, market_value / price), 6)
        market_value = quantity * price
        gain = rng.gauss(float(p["return_bias"]) * 10, float(p["vol"]) * 3)
        cost_price = round(price / max(0.20, 1 + gain), 2)
        cost_value = quantity * cost_price
        profit_loss = market_value - cost_value
        holding_days = max(1, int(rng.gauss(int(p["holding_days"]), max(2, int(p["holding_days"]) * 0.25))))
        last_trade = SNAPSHOT - timedelta(days=min(holding_days, 365))
        symbol = SYMBOLS[(index * 7 + stock_no) % len(SYMBOLS)]
        positions.append({"symbol": symbol, "industry": industry, "quantity": quantity, "price": price,
                          "market_value": market_value, "cost_price": cost_price, "cost_value": cost_value,
                          "profit_loss": profit_loss, "holding_days": holding_days, "last_trade": last_trade})
        industry_totals.setdefault(industry, [0.0, 0.0])
        industry_totals[industry][0] += market_value
        industry_totals[industry][1] += cost_value

    total_stock = sum(x["market_value"] for x in positions)
    stock_rows = [(user_id, _date_text(SNAPSHOT), x["symbol"], x["industry"], f"{x['quantity']:.6f}",
                   _price(x["price"]), _money(x["market_value"]), _price(x["cost_price"]),
                   _money(x["cost_value"]), x["market_value"] / total_stock,
                   _signed_money(x["profit_loss"]), x["profit_loss"] / x["cost_value"], x["holding_days"],
                   _date_text(x["last_trade"])) for x in positions]

    industry_rows = []
    for rank, (industry, (market_value, cost_value)) in enumerate(sorted(industry_totals.items(), key=lambda item: item[1][0], reverse=True), 1):
        pnl = market_value - cost_value
        industry_rows.append((user_id, _date_text(SNAPSHOT), industry, _money(market_value), market_value / total_stock,
                              _money(cost_value), _signed_money(pnl), pnl / cost_value, rank))

    financing = investable * float(p["leverage"]) * rng.uniform(0.75, 1.10)
    lending = investable * (0.04 if latent.event_driven else 0.015) * rng.uniform(0.4, 1.2)
    collateral = total_stock + cash
    credit_limit = max(1_000_000.0, financing / max(0.15, float(p["leverage"]) or 0.15) * 1.25)
    available_credit = max(0.0, credit_limit - financing - lending)
    warning = 1.30
    liquidation = 1.10
    initial_margin_ratio = collateral / max(1.0, financing + lending)
    pressure = initial_margin_ratio < warning + (0.10 if latent.primary == "high_leverage" else 0.0)
    account_rows = [(account_id, user_id, "margin", "active", _money(cash))]
    holding_rows = [(account_id, x["symbol"], f"{x['quantity']:.6f}", _price(x["cost_price"]),
                    f"{SNAPSHOT.isoformat()}T00:00:00.000000Z") for x in positions]

    tx_count = 3 if int(p["holding_days"]) > 150 else (12 if int(p["holding_days"]) < 20 else 6)
    if latent.event_driven:
        tx_count += 3
    tx_rows = []
    for tx_no in range(tx_count):
        pos = positions[tx_no % len(positions)]
        trade_day = SNAPSHOT - timedelta(days=min(55, tx_no * max(1, int(p["holding_days"]) // 8 + 1)))
        side = "buy" if tx_no % 3 else "sell"
        tx_rows.append((f"syn-tx-{index + 1:04d}-{tx_no + 1:03d}", account_id, pos["symbol"], side,
                        f"{max(1.0, pos['quantity'] * rng.uniform(0.03, 0.18)):.6f}", _price(pos["price"]),
                        f"{trade_day.isoformat()}T09:30:00.000000Z"))

    daily_rows = []
    equity = total_stock + cash
    starting_equity = equity
    financing_today = financing
    lending_today = lending
    daily_returns: list[float] = []
    max_equity = equity
    max_drawdown = 0.0
    for day_no in range(TRADING_DAYS):
        trade_day = SNAPSHOT - timedelta(days=TRADING_DAYS - day_no - 1)
        daily_return = rng.gauss(float(p["return_bias"]) / 20, float(p["vol"]))
        if latent.momentum and day_no > TRADING_DAYS // 2:
            daily_return += 0.0015
        daily_pnl = equity * daily_return
        equity = max(1.0, equity + daily_pnl)
        max_equity = max(max_equity, equity)
        max_drawdown = min(max_drawdown, (equity - max_equity) / max_equity)
        collateral_day = max(1.0, starting_equity * (equity / starting_equity))
        if pressure and day_no == TRADING_DAYS - 1:
            financing_today *= 0.92
        ratio = collateral_day / max(1.0, financing_today + lending_today)
        daily_rows.append((user_id, _date_text(trade_day), _money(collateral_day), _money(financing_today),
                           _money(lending_today), _money(cash), ratio, (financing_today + lending_today) / credit_limit,
                           _signed_money(daily_pnl), daily_return, int(ratio < warning), int(ratio < liquidation),
                           _money(50_000 if ratio < warning else 0), _money(financing - financing_today)))
        daily_returns.append(daily_return)

    mean_return = sum(daily_returns) / len(daily_returns)
    volatility = (sum((x - mean_return) ** 2 for x in daily_returns) / len(daily_returns)) ** 0.5
    period_return = equity / max(1.0, total_stock + cash) - 1
    annualized = (1 + period_return) ** (252 / TRADING_DAYS) - 1 if period_return > -1 else -1
    pnl = equity - (total_stock + cash)
    sharpe = mean_return / volatility * (252 ** 0.5) if volatility else 0.0
    turnover = tx_count * rng.uniform(0.08, 0.22) / max(0.1, float(p["holding_days"]) / 30)
    avg_holding = sum(x["holding_days"] for x in positions) / len(positions)
    snapshot_collateral = float(daily_rows[-1][2])
    snapshot_financing = float(daily_rows[-1][3])
    snapshot_lending = float(daily_rows[-1][4])
    snapshot_margin_ratio = float(daily_rows[-1][6])
    snapshot_credit_utilization = float(daily_rows[-1][7])
    snapshot_available_credit = max(0.0, credit_limit - snapshot_financing - snapshot_lending)
    report = (user_id, _date_text(SNAPSHOT), _date_text(SNAPSHOT - timedelta(days=TRADING_DAYS)), _date_text(SNAPSHOT),
              period_return, annualized, _signed_money(pnl), volatility * (252 ** 0.5), max_drawdown, sharpe,
              max(0.0, min(1.0, 0.5 + mean_return * 10)), turnover, avg_holding,
              max(0.0, min(1.0, 0.5 + mean_return * 12)), (snapshot_financing + snapshot_lending) / max(1.0, investable),
              period_return - 0.02)
    market_beta = 0.7 + risk_score * 0.8
    factors = (user_id, _date_text(SNAPSHOT), market_beta, rng.uniform(-0.4, 0.4),
               0.9 if latent.value_quality else rng.uniform(-0.3, 0.3),
               0.9 if latent.growth else rng.uniform(-0.3, 0.3),
               0.9 if latent.momentum else (-0.7 if latent.contrarian else rng.uniform(-0.3, 0.3)),
               0.8 if latent.value_quality else rng.uniform(-0.2, 0.4),
               min(2.5, volatility * 40), rng.uniform(-0.3, 0.4),
               min(2.5, financing / max(1.0, investable) * 3),
               min(2.5, max(x["market_value"] for x in positions) / total_stock * 3))

    age_band = rng.choice(("30-39", "40-49", "50-59", "60-69"))
    region = rng.choice(("synthetic-east", "synthetic-north", "synthetic-south", "synthetic-west"))
    customer_tier = "family_office" if asset >= 10_000_000 else "private_banking"
    base = (user_id, f"Synthetic Client {index + 1:04d}", age_band,
            "high" if risk_score > 0.65 else ("low" if risk_score < 0.35 else "medium"), region,
            customer_tier, f"{2026 - max(1, int(rng.uniform(3, 18)))}-01-01T00:00:00.000000Z")
    query = (user_id, _date_text(SNAPSHOT), _money(investable), _money(asset), _money(cash), risk_score,
             max(1, int(rng.uniform(3, 18))), "long" if avg_holding > 120 else "medium", 0.7 if cash < asset * 0.15 else 0.3,
             min(90, max(2, tx_count * 4)))
    margin = (user_id, _date_text(SNAPSHOT), account_id, _money(credit_limit), _money(snapshot_financing), _money(snapshot_lending),
              _money(snapshot_collateral), _money(cash), _money(snapshot_available_credit), snapshot_margin_ratio, warning, liquidation,
              snapshot_credit_utilization, int(snapshot_margin_ratio < warning), int(snapshot_margin_ratio < liquidation))

    summary = {
        "user_id": user_id, "total_asset": round(asset, 2), "cash_ratio": round(cash / asset, 3),
        "stock_count": len(positions), "max_stock_weight": round(max(x["market_value"] for x in positions) / total_stock, 3),
        "max_industry_weight": round(max(x[0] for x in industry_totals.values()) / total_stock, 3),
        "credit_utilization": round(snapshot_credit_utilization, 3),
        "maintenance_margin_ratio": round(snapshot_margin_ratio, 3), "period_return": round(period_return, 4),
        "annualized_return": round(annualized, 4), "max_drawdown": round(max_drawdown, 4),
        "turnover_rate": round(turnover, 3), "average_holding_days": round(avg_holding, 1),
        "factor_momentum": round(float(factors[6]), 3), "factor_value": round(float(factors[4]), 3),
    }
    explanations = {"user_id": user_id, "summary": summary,
                    "evidence": [
                        f"cash_ratio={summary['cash_ratio']:.3f}",
                        f"max_stock_weight={summary['max_stock_weight']:.3f}",
                        f"max_industry_weight={summary['max_industry_weight']:.3f}",
                        f"credit_utilization={summary['credit_utilization']:.3f}",
                        f"maintenance_margin_ratio={summary['maintenance_margin_ratio']:.3f}",
                        f"turnover_rate={summary['turnover_rate']:.3f}",
                        f"average_holding_days={summary['average_holding_days']:.1f}",
                        f"max_drawdown={summary['max_drawdown']:.4f}",
                    ]}
    return [base, query, margin, report, daily_rows, industry_rows, stock_rows, factors, account_rows, holding_rows, tx_rows], (summary, explanations)


def _build_rows_v2(rng: random.Random, index: int, latent: LatentProfile) -> tuple[list[tuple], dict[str, Any]]:
    """Build one user with a daily PnL bridge and an explicit summary ledger."""
    p = _parameters(latent)
    user_id = f"syn-user-{index + 1:04d}"
    account_id = f"syn-margin-{index + 1:04d}"
    asset = float(p["asset"]) * rng.uniform(0.70, 1.45)
    cash = asset * float(p["cash_ratio"]) * rng.uniform(0.85, 1.15)
    investable = asset - cash * 0.25
    risk_score = min(0.99, max(0.05, float(p["risk"]) + rng.uniform(-0.06, 0.06)))
    industries = list(INDUSTRIES)
    rng.shuffle(industries)
    count = int(p["stock_count"])
    if latent.primary == "single_stock_concentrated":
        weights = [float(p["concentration"])] + [rng.uniform(0.02, 0.08) for _ in range(count - 1)]
    elif latent.primary == "industry_concentrated":
        weights = [rng.uniform(0.04, 0.12) for _ in range(count)]
        weights[0] += float(p["concentration"])
    else:
        weights = [rng.uniform(0.6, 1.4) for _ in range(count)]
    weights = [w / sum(weights) for w in weights]
    if latent.primary == "single_stock_concentrated":
        weights[0] = float(p["concentration"])
        rest = 1.0 - weights[0]
        tail = sum(weights[1:])
        weights[1:] = [w / tail * rest for w in weights[1:]]
    elif latent.primary == "industry_concentrated":
        weights[0] = min(0.75, 0.42 + rng.uniform(0.0, 0.12))
        rest = 1.0 - weights[0]
        tail = sum(weights[1:])
        weights[1:] = [w / tail * rest for w in weights[1:]]

    positions: list[dict[str, Any]] = []
    industry_totals: dict[str, list[float]] = {}
    for stock_no, weight in enumerate(weights):
        industry = industries[0] if latent.primary == "industry_concentrated" and stock_no < max(2, count // 2) else industries[stock_no % len(industries)]
        price = round(rng.uniform(25, 420), 2)
        quantity = round(max(1.0, investable * 0.78 * weight / price), 6)
        market_value = quantity * price
        gain = rng.gauss(float(p["return_bias"]) * 10, float(p["vol"]) * 3)
        cost_price = round(price / max(0.20, 1 + gain), 2)
        cost_value = quantity * cost_price
        profit_loss = market_value - cost_value
        holding_days = max(1, int(rng.gauss(int(p["holding_days"]), max(2, int(p["holding_days"]) * 0.25))))
        positions.append({
            "symbol": SYMBOLS[(index * 7 + stock_no) % len(SYMBOLS)], "industry": industry,
            "quantity": quantity, "price": price, "market_value": market_value,
            "cost_price": cost_price, "cost_value": cost_value, "profit_loss": profit_loss,
            "holding_days": holding_days, "last_trade": SNAPSHOT - timedelta(days=min(holding_days, 365)),
        })
        industry_totals.setdefault(industry, [0.0, 0.0])
        industry_totals[industry][0] += market_value
        industry_totals[industry][1] += cost_value

    total_stock = sum(x["market_value"] for x in positions)
    stock_rows = [(
        user_id, _date_text(SNAPSHOT), x["symbol"], x["industry"], f"{x['quantity']:.6f}",
        _price(x["price"]), _money(x["market_value"]), _price(x["cost_price"]), _money(x["cost_value"]),
        x["market_value"] / total_stock, _signed_money(x["profit_loss"]),
        x["profit_loss"] / x["cost_value"], x["holding_days"], _date_text(x["last_trade"]),
    ) for x in positions]
    industry_rows = []
    for rank, (industry, (market_value, cost_value)) in enumerate(sorted(industry_totals.items(), key=lambda item: item[1][0], reverse=True), 1):
        pnl = market_value - cost_value
        industry_rows.append((
            user_id, _date_text(SNAPSHOT), industry, _money(market_value), market_value / total_stock,
            _money(cost_value), _signed_money(pnl), pnl / cost_value, rank,
        ))

    financing = investable * float(p["leverage"]) * rng.uniform(0.75, 1.10)
    lending = investable * (0.04 if latent.event_driven else 0.015) * rng.uniform(0.4, 1.2)
    credit_limit = max(1_000_000.0, financing / max(0.15, float(p["leverage"]) or 0.15) * 1.25)
    warning, liquidation = 1.30, 1.10
    starting_equity = total_stock + cash
    beginning_unrealized = rng.gauss(0.0, total_stock * 0.040)
    ending_unrealized = sum(x["profit_loss"] for x in positions)
    tx_count = 3 if int(p["holding_days"]) > 150 else (12 if int(p["holding_days"]) < 20 else 6)
    if latent.event_driven:
        tx_count += 3
    turnover = tx_count * rng.uniform(0.08, 0.22) / max(0.1, float(p["holding_days"]) / 30)
    financing_interest = financing * rng.uniform(0.00010, 0.00018) * TRADING_DAYS
    securities_fee = lending * rng.uniform(0.00012, 0.00022) * TRADING_DAYS
    transaction_fees = total_stock * turnover * rng.uniform(0.00015, 0.00035)
    other_fees = total_stock * rng.uniform(0.00002, 0.00008)
    total_costs = financing_interest + securities_fee + transaction_fees + other_fees
    realized_target = rng.gauss(
        total_stock * float(p["return_bias"]) * 0.20,
        total_stock * (0.025 if int(p["holding_days"]) < 30 else 0.015),
    )

    raw_unrealized = [rng.gauss((ending_unrealized - beginning_unrealized) / TRADING_DAYS, total_stock * float(p["vol"]) * 0.60) for _ in range(TRADING_DAYS)]
    correction = (ending_unrealized - beginning_unrealized - sum(raw_unrealized)) / TRADING_DAYS
    unrealized_changes = [value + correction for value in raw_unrealized]
    raw_realized = [rng.gauss(realized_target / TRADING_DAYS, abs(realized_target) * 0.10 / TRADING_DAYS + total_stock * 0.0002) for _ in range(TRADING_DAYS)]
    realized_correction = (realized_target - sum(raw_realized)) / TRADING_DAYS
    realized_changes = [value + realized_correction for value in raw_realized]
    daily_costs = [total_costs / TRADING_DAYS for _ in range(TRADING_DAYS)]

    account_rows = [(account_id, user_id, "margin", "active", _money(cash))]
    holding_rows = [(account_id, x["symbol"], f"{x['quantity']:.6f}", _price(x["cost_price"]), f"{SNAPSHOT.isoformat()}T00:00:00.000000Z") for x in positions]
    tx_rows = []
    for tx_no in range(tx_count):
        pos = positions[tx_no % len(positions)]
        trade_day = SNAPSHOT - timedelta(days=min(55, tx_no * max(1, int(p["holding_days"]) // 8 + 1)))
        tx_rows.append((
            f"syn-tx-{index + 1:04d}-{tx_no + 1:03d}", account_id, pos["symbol"], "buy" if tx_no % 3 else "sell",
            f"{max(1.0, pos['quantity'] * rng.uniform(0.03, 0.18)):.6f}", _price(pos["price"]),
            f"{trade_day.isoformat()}T09:30:00.000000Z",
        ))

    daily_rows = []
    equity = starting_equity
    daily_returns: list[float] = []
    max_equity = equity
    max_drawdown = 0.0
    for day_no in range(TRADING_DAYS):
        trade_day = SNAPSHOT - timedelta(days=TRADING_DAYS - day_no - 1)
        daily_pnl = realized_changes[day_no] + unrealized_changes[day_no] - daily_costs[day_no]
        previous_equity = equity
        equity = max(1.0, equity + daily_pnl)
        daily_return = daily_pnl / max(1.0, previous_equity)
        daily_returns.append(daily_return)
        max_equity = max(max_equity, equity)
        max_drawdown = min(max_drawdown, (equity - max_equity) / max_equity)
        deleveraging = financing * 0.08 * (day_no / max(1, TRADING_DAYS - 1)) if latent.primary == "high_leverage" and day_no > TRADING_DAYS * 0.65 else 0.0
        financing_today = financing - deleveraging
        lending_today = lending
        collateral_day = max(1.0, equity)
        ratio = collateral_day / max(1.0, financing_today + lending_today)
        daily_rows.append((
            user_id, _date_text(trade_day), _money(collateral_day), _money(financing_today), _money(lending_today), _money(cash),
            ratio, (financing_today + lending_today) / credit_limit, _signed_money(realized_changes[day_no]),
            _signed_money(unrealized_changes[day_no]), _money(daily_costs[day_no]), _signed_money(daily_pnl), daily_return,
            int(ratio < warning), int(ratio < liquidation), _money(50_000 if ratio < warning else 0), _money(deleveraging),
        ))

    period_net_pnl = sum(float(row[11]) for row in daily_rows)
    portfolio_return = period_net_pnl / starting_equity
    volatility = (sum((x - sum(daily_returns) / len(daily_returns)) ** 2 for x in daily_returns) / len(daily_returns)) ** 0.5
    annualized = (1 + portfolio_return) ** (252 / TRADING_DAYS) - 1 if portfolio_return > -1 else -1
    sharpe = (sum(daily_returns) / len(daily_returns)) / volatility * (252 ** 0.5) if volatility else 0.0
    avg_holding = sum(x["holding_days"] for x in positions) / len(positions)
    last_daily = daily_rows[-1]
    snapshot_financing, snapshot_lending = float(last_daily[3]), float(last_daily[4])
    report = (
        user_id, _date_text(SNAPSHOT), _date_text(SNAPSHOT - timedelta(days=TRADING_DAYS)), _date_text(SNAPSHOT),
        _money(starting_equity), _signed_money(period_net_pnl), _signed_money(realized_target), _signed_money(beginning_unrealized),
        _signed_money(ending_unrealized), _money(financing_interest), _money(securities_fee), _money(transaction_fees), _money(other_fees),
        portfolio_return, annualized, volatility * (252 ** 0.5), max_drawdown, sharpe,
        max(0.0, min(1.0, 0.5 + sum(daily_return > 0 for daily_return in daily_returns) / len(daily_returns) - 0.5)),
        turnover, avg_holding, max(0.0, min(1.0, 0.5 + realized_target / max(1.0, total_stock) * 10)),
        (snapshot_financing + snapshot_lending) / max(1.0, investable), portfolio_return - 0.02,
    )
    risk_level = "high" if risk_score > 0.65 else ("low" if risk_score < 0.35 else "medium")
    factors = (
        user_id, _date_text(SNAPSHOT), 0.7 + risk_score * 0.8, rng.uniform(-0.4, 0.4),
        0.9 if latent.value_quality else rng.uniform(-0.3, 0.3), 0.9 if latent.growth else rng.uniform(-0.3, 0.3),
        0.9 if latent.momentum else (-0.7 if latent.contrarian else rng.uniform(-0.3, 0.3)),
        0.8 if latent.value_quality else rng.uniform(-0.2, 0.4), min(2.5, volatility * 40), rng.uniform(-0.3, 0.4),
        min(2.5, snapshot_financing / max(1.0, investable) * 3), min(2.5, max(x["market_value"] for x in positions) / total_stock * 3),
    )
    region = rng.choice(("synthetic-east", "synthetic-north", "synthetic-south", "synthetic-west"))
    customer_tier = "family_office" if asset >= 10_000_000 else "private_banking"
    max_risk = "R5" if risk_score >= 0.75 else ("R4" if risk_score >= 0.55 else ("R3" if risk_score >= 0.35 else "R2"))
    margin_enabled = latent.primary != "cash_defensive" or financing > 0
    short_selling_enabled = lending > 0 and risk_score >= 0.55
    base = (user_id, f"Synthetic Client {index + 1:04d}", rng.choice(("30-39", "40-49", "50-59", "60-69")), risk_level, region, customer_tier, f"{2026 - max(1, int(rng.uniform(3, 18)))}-01-01T00:00:00.000000Z")
    bucket = _asset_bucket(asset)
    query = (
        user_id, _date_text(SNAPSHOT), _money(investable), _money(asset), _money(cash), risk_score,
        max(1, int(rng.uniform(3, 18))), "long" if avg_holding > 120 else "medium", 0.7 if cash < asset * 0.15 else 0.3,
        min(90, max(2, tx_count * 4)), 1, int(margin_enabled), int(short_selling_enabled), max_risk, bucket,
    )
    margin = (
        user_id, _date_text(SNAPSHOT), account_id, _money(credit_limit), _money(snapshot_financing), _money(snapshot_lending),
        _money(float(last_daily[2])), _money(cash), _money(max(0.0, credit_limit - snapshot_financing - snapshot_lending)),
        float(last_daily[6]), warning, liquidation, float(last_daily[7]), int(last_daily[13]), int(last_daily[14]),
    )
    summary = {
        "user_id": user_id, "total_asset": round(asset, 2), "cash_ratio": round(cash / asset, 3), "stock_count": len(positions),
        "max_stock_weight": round(max(x["market_value"] for x in positions) / total_stock, 3),
        "max_industry_weight": round(max(x[0] for x in industry_totals.values()) / total_stock, 3),
        "credit_utilization": round(float(last_daily[7]), 3), "maintenance_margin_ratio": round(float(last_daily[6]), 3),
        "period_return": round(portfolio_return, 4), "annualized_return": round(annualized, 4), "max_drawdown": round(max_drawdown, 4),
        "turnover_rate": round(turnover, 3), "average_holding_days": round(avg_holding, 1),
        "factor_momentum": round(float(factors[6]), 3), "factor_value": round(float(factors[4]), 3),
    }
    explanations = {"user_id": user_id, "summary": summary, "evidence": [f"cash_ratio={summary['cash_ratio']:.3f}", f"max_stock_weight={summary['max_stock_weight']:.3f}", f"max_industry_weight={summary['max_industry_weight']:.3f}", f"credit_utilization={summary['credit_utilization']:.3f}", f"maintenance_margin_ratio={summary['maintenance_margin_ratio']:.3f}", f"turnover_rate={summary['turnover_rate']:.3f}", f"average_holding_days={summary['average_holding_days']:.1f}", f"max_drawdown={summary['max_drawdown']:.4f}"]}
    return [base, query, margin, report, daily_rows, industry_rows, stock_rows, factors, account_rows, holding_rows, tx_rows], (summary, explanations)


def generate_synthetic_data(
    path: Path, *, seed: int = DEFAULT_SEED, overwrite: bool = False, user_count: int = USER_COUNT,
) -> GenerationResult:
    """Generate a deterministic synthetic margin-user database atomically."""
    if user_count < 20:
        raise ValueError("Synthetic dataset requires at least 20 users")
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Database exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    latent_rng = random.Random(seed ^ 0x5A5A)
    primaries = _allocate_primaries(user_count, seed)
    generated: list[tuple[list[tuple], dict[str, Any]]] = []
    for index, primary in enumerate(primaries):
        generated.append(_build_rows_v2(rng, index, _latent(primary, latent_rng)))

    fd, temporary = tempfile.mkstemp(prefix=f".synthetic-{user_count}-", suffix=".db", dir=path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(DDL)
            with connection:
                for rows, _ in generated:
                    base, query, margin, report, daily, industries, stocks, factors, accounts, holdings, transactions = rows
                    connection.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", base)
                    connection.execute("INSERT INTO query_base_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", query)
                    connection.execute("INSERT INTO margin_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", margin)
                    connection.execute("INSERT INTO report_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", report)
                    connection.executemany("INSERT INTO margin_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", daily)
                    connection.executemany("INSERT INTO industry_position VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", industries)
                    connection.executemany("INSERT INTO stock_position VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", stocks)
                    connection.execute("INSERT INTO factors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", factors)
                    connection.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", accounts[0])
                    connection.executemany("INSERT INTO holdings VALUES (?, ?, ?, ?, ?)", holdings)
                    connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)", transactions)
                persisted = connection.execute("""
                    SELECT r.user_id, u.region, q.asset_bucket, r.portfolio_return,
                           r.benchmark_excess_return, r.max_drawdown, r.sharpe_ratio
                    FROM report_metrics r JOIN users u USING(user_id)
                    JOIN query_base_info q USING(user_id)
                """).fetchall()
                groups: dict[str, list[sqlite3.Row]] = {}
                for row in persisted:
                    groups.setdefault(f"{row['region']}|{row['asset_bucket']}", []).append(row)

                def competition_ranks(rows: list[sqlite3.Row], field: str) -> dict[str, int]:
                    ordered = sorted(rows, key=lambda row: row[field], reverse=True)
                    result: dict[str, int] = {}
                    previous = object()
                    current_rank = 0
                    for position, row in enumerate(ordered, 1):
                        if row[field] != previous:
                            current_rank = position
                            previous = row[field]
                        result[row["user_id"]] = current_rank
                    return result

                ranks: dict[str, tuple[str, int, int, int, int]] = {}
                for group_key, rows in groups.items():
                    return_ranks = competition_ranks(rows, "portfolio_return")
                    drawdown_ranks = competition_ranks(rows, "max_drawdown")
                    sharpe_ranks = competition_ranks(rows, "sharpe_ratio")
                    for row in rows:
                        ranks[row["user_id"]] = (group_key, len(rows), return_ranks[row["user_id"]], drawdown_ranks[row["user_id"]], sharpe_ranks[row["user_id"]])
                for row in persisted:
                    group_key, group_size, return_rank, drawdown_rank, risk_adjusted_rank = ranks[row["user_id"]]
                    percentile = (group_size - return_rank) / (group_size - 1) if group_size > 1 else 1.0
                    connection.execute(
                        "INSERT INTO user_return_rank VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (row["user_id"], _date_text(SNAPSHOT), group_key, group_size,
                         row["portfolio_return"], row["benchmark_excess_return"], return_rank, percentile,
                         drawdown_rank, risk_adjusted_rank),
                    )
        finally:
            connection.close()
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

    summaries = [item[1][0] for item in generated]
    explanations = [item[1][1] for item in generated]
    constraints = [
        f"users = {user_count} and user_id is unique",
        "stock_position market_value = quantity * market_price; weight sums to approximately 1 per user",
        "industry_position is the aggregation of stock_position by industry",
        "credit_utilization = (financing_balance + securities_lending_balance) / credit_limit",
        "available_credit = credit_limit - financing_balance - securities_lending_balance",
        "maintenance_margin_ratio = collateral_market_value / (financing_balance + securities_lending_balance)",
        "report_metrics and user_return_rank are derived from generated daily/per-position values",
        "no archetype/user_type/personality field is present in the database schema",
    ]
    selected_indexes = [int((user_count - 1) * quantile) for quantile in (0.08, 0.24, 0.50, 0.72, 0.91)]
    selected = [explanations[index] for index in selected_indexes]
    distribution = {name: primaries.count(name) for name in PRIMARY_WEIGHTS}
    return GenerationResult(path, seed, summaries, selected, constraints, distribution)
