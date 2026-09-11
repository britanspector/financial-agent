"""Business-level contracts for the synthetic margin-user tools."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from financial_agent.schemas import Schema

Amount = Annotated[Decimal, Field(allow_inf_nan=False)]
NonNegativeAmount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
RiskLevel = Literal["R2", "R3", "R4", "R5"]


class UserInput(Schema):
    user_id: str = Field(min_length=1, max_length=128)


class MarginAccountInput(UserInput):
    start_date: date | None = Field(default=None, description="Inclusive first date of daily history")
    end_date: date | None = Field(default=None, description="Exclusive end date of daily history")
    limit: int = Field(default=60, ge=1, le=366, strict=True)
    offset: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.start_date is not None and self.end_date is not None and self.start_date >= self.end_date:
            raise ValueError("start_date must precede end_date")
        return self


class CustomerContext(Schema):
    user_id: str
    name_alias: str
    age_band: str | None
    risk_level: str | None
    region: str | None
    customer_tier: str
    created_at: AwareDatetime
    snapshot_date: str
    asset_bucket: str
    investable_asset: NonNegativeAmount
    total_asset: NonNegativeAmount
    cash_asset: NonNegativeAmount
    risk_tolerance_score: float = Field(ge=0, le=1)
    investment_experience_years: int = Field(ge=0)
    investment_horizon: str
    liquidity_need_score: float = Field(ge=0, le=1)
    active_trading_days_90d: int = Field(ge=0, le=90)
    trade_enabled: bool
    margin_enabled: bool
    short_selling_enabled: bool
    max_allowed_product_risk_level: RiskLevel


class MarginInfo(Schema):
    user_id: str
    snapshot_date: str
    margin_account_id: str
    credit_limit: NonNegativeAmount
    financing_balance: NonNegativeAmount
    securities_lending_balance: NonNegativeAmount
    collateral_market_value: NonNegativeAmount
    cash_balance: NonNegativeAmount
    available_credit: NonNegativeAmount
    maintenance_margin_ratio: float = Field(ge=0)
    warning_line: float = Field(ge=0)
    liquidation_line: float = Field(ge=0)
    credit_utilization: float = Field(ge=0, le=1)
    margin_call_flag: bool
    forced_liquidation_flag: bool


class MarginDaily(Schema):
    user_id: str
    trade_date: str
    collateral_market_value: NonNegativeAmount
    financing_balance: NonNegativeAmount
    securities_lending_balance: NonNegativeAmount
    cash_balance: NonNegativeAmount
    maintenance_margin_ratio: float = Field(ge=0)
    credit_utilization: float = Field(ge=0, le=1)
    daily_realized_pnl: Amount
    daily_unrealized_pnl_change: Amount
    daily_total_costs: NonNegativeAmount
    daily_profit_loss: Amount
    daily_return: float
    margin_call_flag: bool
    forced_liquidation_flag: bool
    collateral_injection_amount: NonNegativeAmount
    deleveraging_amount: NonNegativeAmount


class MarginAccount(Schema):
    user_id: str
    info: MarginInfo
    daily: list[MarginDaily]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=366)
    offset: int = Field(ge=0)
    start_date: str | None
    end_date: str | None


class IndustryPosition(Schema):
    user_id: str
    snapshot_date: str
    industry_code: str
    market_value: NonNegativeAmount
    weight: float = Field(ge=0, le=1)
    cost_value: NonNegativeAmount
    profit_loss: Amount
    profit_loss_pct: float
    position_rank: int = Field(ge=1)


class StockPosition(Schema):
    user_id: str
    snapshot_date: str
    stock_code: str
    industry_code: str
    quantity: NonNegativeAmount
    market_price: NonNegativeAmount
    market_value: NonNegativeAmount
    cost_price: NonNegativeAmount
    cost_value: NonNegativeAmount
    weight: float = Field(ge=0, le=1)
    profit_loss: Amount
    profit_loss_pct: float
    holding_days: int = Field(ge=1)
    last_trade_date: str


class PortfolioPositions(Schema):
    user_id: str
    snapshot_date: str
    currency: Literal["CNY"] = "CNY"
    industries: list[IndustryPosition]
    stocks: list[StockPosition]


class ReportMetrics(Schema):
    user_id: str
    report_date: str
    period_start: str
    period_end: str
    beginning_equity: NonNegativeAmount
    period_net_pnl: Amount
    realized_pnl: Amount
    beginning_unrealized_pnl: Amount
    ending_unrealized_pnl: Amount
    financing_interest: NonNegativeAmount
    securities_lending_fee: NonNegativeAmount
    transaction_fees: NonNegativeAmount
    other_fees: NonNegativeAmount
    portfolio_return: float
    annualized_return: float
    volatility: float = Field(ge=0)
    max_drawdown: float = Field(le=0)
    sharpe_ratio: float
    win_rate: float = Field(ge=0, le=1)
    turnover_rate: float = Field(ge=0)
    average_holding_days: float = Field(gt=0)
    profit_trade_ratio: float = Field(ge=0, le=1)
    margin_contribution: float
    benchmark_excess_return: float


class Factors(Schema):
    user_id: str
    snapshot_date: str
    market_beta: float
    size_exposure: float
    value_exposure: float
    growth_exposure: float
    momentum_exposure: float
    quality_exposure: float
    volatility_exposure: float
    liquidity_exposure: float
    leverage_exposure: float = Field(ge=0)
    concentration_exposure: float = Field(ge=0)


class UserReturnRank(Schema):
    user_id: str
    rank_date: str
    peer_group_key: str
    peer_group_size: int = Field(ge=1)
    period_return: float
    benchmark_excess_return: float
    return_rank: int = Field(ge=1)
    return_percentile: float = Field(ge=0, le=1)
    drawdown_rank: int = Field(ge=1)
    risk_adjusted_rank: int = Field(ge=1)


class PortfolioAnalytics(Schema):
    user_id: str
    report: ReportMetrics
    factors: Factors
    rank: UserReturnRank
