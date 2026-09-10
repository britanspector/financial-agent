"""Stable Pydantic contracts for market data tools."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from financial_agent.schemas import Schema

AssetType = Literal["stock", "index"]
Adjustment = Literal["qfq", "hfq", "none"]
CanonicalSymbol = str
Amount = Annotated[Decimal, Field(allow_inf_nan=False)]
NonNegativeAmount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class MarketSnapshotInput(Schema):
    symbol: str = Field(min_length=1, max_length=32)


class MarketHistoryInput(Schema):
    symbol: str = Field(min_length=1, max_length=32)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def valid_interval(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must precede end_date")
        return self


class MarketSnapshot(Schema):
    symbol: str
    snapshot_kind: Literal["daily_close"]
    as_of: AwareDatetime
    price: NonNegativeAmount
    open: NonNegativeAmount
    high: NonNegativeAmount
    low: NonNegativeAmount
    previous_close: NonNegativeAmount
    pct_change: Amount
    volume: NonNegativeAmount
    amount: NonNegativeAmount
    source: Literal["tushare"]


class MarketBar(Schema):
    trade_date: date
    open: NonNegativeAmount
    high: NonNegativeAmount
    low: NonNegativeAmount
    close: NonNegativeAmount
    previous_close: NonNegativeAmount | None
    pct_change: Amount
    volume: NonNegativeAmount
    amount: NonNegativeAmount
    source: Literal["tushare"]


class MarketHistory(Schema):
    symbol: str
    asset_type: Literal["stock"]
    start_date: date
    end_date: date
    adjustment: Literal["none"]
    bars: list[MarketBar]
    source: Literal["tushare"]
