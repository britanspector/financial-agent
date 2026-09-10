"""Synchronous adapter for the official Tushare REST API."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from financial_agent.market_data.models import (
    CanonicalSymbol,
    MarketBar,
    MarketHistory,
    MarketSnapshot,
)
from financial_agent.market_data.provider import (
    EmptyResultError,
    ProviderAdapterError,
    ProviderError,
    ProviderPermissionDeniedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
_REQUIRED_FIELDS = frozenset(_FIELDS.split(","))
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class TushareProvider:
    """Convert Tushare REST rows directly into stable domain models."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.tushare.pro",
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        if not token:
            raise ValueError("Tushare token is required")
        self._token = token
        self._base_url = base_url
        self._clock = clock or (lambda: datetime.now(_SHANGHAI))
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TushareProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_snapshot(self, symbol: CanonicalSymbol) -> MarketSnapshot:
        today = self._clock().astimezone(_SHANGHAI).date()
        rows = self._daily_rows(symbol, today - timedelta(days=90), today)
        if not rows:
            raise EmptyResultError("No daily snapshot rows returned")
        row = max(rows, key=_trade_date)
        trade_date = _trade_date(row)
        return MarketSnapshot(
            symbol=symbol,
            snapshot_kind="daily_close",
            as_of=datetime.combine(trade_date, time(15, 0), tzinfo=_SHANGHAI),
            price=_decimal(row, "close"),
            open=_decimal(row, "open"),
            high=_decimal(row, "high"),
            low=_decimal(row, "low"),
            previous_close=_decimal(row, "pre_close"),
            pct_change=_decimal(row, "pct_chg"),
            volume=_decimal(row, "vol") * Decimal(100),
            amount=_decimal(row, "amount") * Decimal(1000),
            source="tushare",
        )

    def get_history(
        self,
        symbol: CanonicalSymbol,
        start_date: date,
        end_date: date,
    ) -> MarketHistory:
        rows = self._daily_rows(symbol, start_date, end_date - timedelta(days=1))
        selected = sorted(
            (row for row in rows if start_date <= _trade_date(row) < end_date),
            key=_trade_date,
        )
        if not selected:
            raise EmptyResultError("No history rows in requested interval")
        bars = [
            MarketBar(
                trade_date=_trade_date(row),
                open=_decimal(row, "open"),
                high=_decimal(row, "high"),
                low=_decimal(row, "low"),
                close=_decimal(row, "close"),
                previous_close=_decimal(row, "pre_close"),
                pct_change=_decimal(row, "pct_chg"),
                volume=_decimal(row, "vol") * Decimal(100),
                amount=_decimal(row, "amount") * Decimal(1000),
                source="tushare",
            )
            for row in selected
        ]
        return MarketHistory(
            symbol=symbol,
            asset_type="stock",
            start_date=start_date,
            end_date=end_date,
            adjustment="none",
            bars=bars,
            source="tushare",
        )

    def _daily_rows(self, symbol: CanonicalSymbol, start_date: date, end_date: date) -> list[dict[str, Any]]:
        payload = {
            "api_name": "daily",
            "token": self._token,
            "params": {
                "ts_code": symbol,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "fields": _FIELDS,
        }
        try:
            response = self._client.post(self._base_url, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException:
            raise ProviderTimeoutError() from None
        except httpx.RequestError:
            raise ProviderUnavailableError() from None
        except httpx.HTTPStatusError:
            raise ProviderError("Tushare returned an unexpected HTTP status") from None

        try:
            body = response.json()
        except ValueError:
            raise ProviderAdapterError("Tushare returned invalid JSON") from None
        if not isinstance(body, dict) or not isinstance(body.get("code"), int):
            raise ProviderAdapterError("Tushare response envelope is invalid")
        code = body["code"]
        if code == 2002:
            raise ProviderPermissionDeniedError(
                _safe_provider_message(body.get("msg"), "Tushare permission denied", self._token)
            )
        if code != 0:
            raise ProviderError(_safe_provider_message(body.get("msg"), "Tushare returned an error", self._token))
        return _rows_from_data(body.get("data"))


def _rows_from_data(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise ProviderAdapterError("Tushare response data is invalid")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ProviderAdapterError("Tushare response fields are invalid")
    if not _REQUIRED_FIELDS.issubset(fields):
        raise ProviderAdapterError("Tushare response is missing required fields")
    if not isinstance(items, list):
        raise ProviderAdapterError("Tushare response items are invalid")
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, list) or len(item) != len(fields):
            raise ProviderAdapterError("Tushare response row does not match fields")
        rows.append(dict(zip(fields, item, strict=True)))
    return rows


def _trade_date(row: dict[str, Any]) -> date:
    value = row.get("trade_date")
    if not isinstance(value, str):
        raise ProviderAdapterError("Tushare trade_date is invalid")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise ProviderAdapterError("Tushare trade_date is invalid") from None


def _decimal(row: dict[str, Any], field: str) -> Decimal:
    value = row.get(field)
    if value is None or isinstance(value, bool):
        raise ProviderAdapterError(f"Tushare {field} is invalid")
    if isinstance(value, float) and not isfinite(value):
        raise ProviderAdapterError(f"Tushare {field} is invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ProviderAdapterError(f"Tushare {field} is invalid") from None
    if not result.is_finite():
        raise ProviderAdapterError(f"Tushare {field} is invalid")
    return result


def _safe_provider_message(value: Any, fallback: str, token: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.replace(token, "[REDACTED]").strip()[:300]
