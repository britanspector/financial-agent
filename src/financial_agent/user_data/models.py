"""Data contracts shared by repositories, services, and tools."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from financial_agent.schemas import Schema

Amount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class Profile(Schema):
    user_id: str
    name_alias: str
    age_band: str | None
    risk_level: str | None
    region: str | None
    customer_tier: str
    created_at: AwareDatetime


class Holding(Schema):
    symbol: str
    quantity: Amount
    avg_cost: Amount
    updated_at: AwareDatetime


class Account(Schema):
    account_id: str
    user_id: str
    account_type: str
    status: Literal["active", "closed"]
    cash_balance: Amount
    holdings: list[Holding] = Field(default_factory=list)


class Portfolio(Schema):
    user_id: str
    currency: Literal["CNY"] = "CNY"
    accounts: list[Account]


class Transaction(Schema):
    transaction_id: str
    account_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Amount
    price: Amount
    trade_time: AwareDatetime


class TransactionPage(Schema):
    user_id: str
    currency: Literal["CNY"] = "CNY"
    transactions: list[Transaction]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class UserInput(Schema):
    user_id: str = Field(min_length=1, max_length=128)


class TransactionsInput(UserInput):
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    limit: int = Field(default=20, ge=1, le=100, strict=True)
    offset: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("start_time must precede end_time")
        return self
