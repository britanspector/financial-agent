"""Provider abstraction and typed provider failures."""

from datetime import date
from typing import Protocol

from financial_agent.market_data.models import (
    CanonicalSymbol, MarketHistory, MarketSnapshot,
)


class MarketDataError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class EmptyResultError(MarketDataError):
    def __init__(self, message: str = "No market data found"):
        super().__init__("EMPTY_RESULT", message)


class ProviderTimeoutError(MarketDataError):
    def __init__(self, message: str = "Market data provider timed out"):
        super().__init__("PROVIDER_TIMEOUT", message, retryable=True)


class ProviderUnavailableError(MarketDataError):
    def __init__(self, message: str = "Market data provider unavailable"):
        super().__init__("PROVIDER_UNAVAILABLE", message, retryable=True)


class ProviderAdapterError(MarketDataError):
    def __init__(self, message: str = "Market data provider response could not be adapted"):
        super().__init__("PROVIDER_ERROR", message)


class ProviderPermissionDeniedError(MarketDataError):
    def __init__(self, message: str = "Market data provider permission denied"):
        super().__init__("PROVIDER_PERMISSION_DENIED", message)


class ProviderError(MarketDataError):
    def __init__(self, message: str = "Market data provider returned an error"):
        super().__init__("PROVIDER_ERROR", message)


class MarketDataProvider(Protocol):
    def get_snapshot(self, symbol: CanonicalSymbol) -> MarketSnapshot: ...

    def get_history(
        self,
        symbol: CanonicalSymbol,
        start_date: date,
        end_date: date,
    ) -> MarketHistory: ...
