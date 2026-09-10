"""Thin business boundary for market data tools."""

from time import perf_counter
from uuid import UUID, uuid4

from pydantic import ValidationError

from financial_agent.tools.contracts import ToolFailure, ToolResult
from financial_agent.tools.registry import ToolSpec
from financial_agent.user_data.auth import CallContext
from financial_agent.market_data.models import MarketHistoryInput, MarketSnapshotInput
from financial_agent.market_data.normalizer import InvalidSymbolError, normalize
from financial_agent.market_data.provider import MarketDataError, MarketDataProvider


class MarketDataService:
    def __init__(self, provider: MarketDataProvider):
        self._provider = provider

    def execute(
        self,
        spec: ToolSpec | None,
        arguments: object,
        *,
        context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult:
        del context
        started = perf_counter()
        request_id = request_id or uuid4()
        data = error = None
        status = "error"
        try:
            if spec is None:
                raise ToolFailure("UNKNOWN_TOOL", "Unknown tool", 404)
            try:
                parameters = spec.input_model.model_validate(arguments)
            except ValidationError:
                raise ToolFailure("INVALID_ARGUMENT", "Invalid market-data arguments", 422) from None

            if spec.name == "get_market_snapshot":
                if not isinstance(parameters, MarketSnapshotInput):
                    raise ToolFailure("INVALID_ARGUMENT", "Invalid market snapshot arguments", 422)
                canonical = normalize(parameters.symbol, "stock")
                data = self._provider.get_snapshot(canonical)
            elif spec.name == "get_market_history":
                if not isinstance(parameters, MarketHistoryInput):
                    raise ToolFailure("INVALID_ARGUMENT", "Invalid market history arguments", 422)
                canonical = normalize(parameters.symbol, "stock")
                data = self._provider.get_history(
                    canonical, parameters.start_date, parameters.end_date,
                )
            else:
                raise ToolFailure("UNKNOWN_TOOL", "Unknown tool", 404)
            data = spec.output_model.model_validate(data)
            status = "success"
        except ToolFailure as exc:
            error = exc.error
        except InvalidSymbolError:
            error = ToolFailure("INVALID_SYMBOL", "Invalid market symbol", 422).error
        except MarketDataError as exc:
            error = ToolFailure(exc.code, exc.message, _status_for(exc.code), retryable=exc.retryable).error
        except Exception:
            error = ToolFailure("INTERNAL_ERROR", "Internal market-data failure", 500).error
        elapsed = max(0.0, (perf_counter() - started) * 1000)
        if error is not None:
            data, status = None, "error"
        result_type = ToolResult[spec.output_model] if spec else ToolResult
        return result_type(
            status=status,
            data=data,
            source="tushare",
            latency=elapsed,
            error=error,
            request_id=request_id,
        )


def _status_for(code: str) -> int:
    return {
        "EMPTY_RESULT": 404,
        "PROVIDER_TIMEOUT": 504,
        "PROVIDER_UNAVAILABLE": 503,
        "PROVIDER_PERMISSION_DENIED": 403,
        "PROVIDER_ERROR": 502,
        "INTERNAL_ERROR": 500,
    }.get(code, 500)
