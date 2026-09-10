"""Async HTTP adapter that restores the Agent ToolResult contract."""

import asyncio
import logging
import threading
from time import perf_counter
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError

from financial_agent.tools.contracts import ToolError, ToolFailure, ToolResult
from financial_agent.tools.registry import ToolSpec
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.http_models import ApiError

logger = logging.getLogger(__name__)


class HttpUserDataClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def execute(self, spec: ToolSpec | None, arguments: object, *, context: CallContext, request_id: UUID | None = None) -> ToolResult:
        request_id = request_id or uuid4()
        return _run_sync(self.async_execute(spec, arguments, context=context, request_id=request_id))

    async def async_execute(
        self, spec: ToolSpec | None, arguments: object, *, context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult:
        started = perf_counter()
        request_id = request_id or uuid4()
        if spec is None:
            return self._error(spec, request_id, started, "UNKNOWN_TOOL", "Unknown tool", 404, False)
        try:
            parameters = spec.input_model.model_validate(arguments)
        except ValidationError:
            return self._error(spec, request_id, started, "INVALID_ARGUMENT", "Invalid tool arguments", 422, False)

        key = context.api_key.get_secret_value() if context.api_key is not None else self.api_key
        headers = {"X-Request-ID": str(request_id)}
        if key:
            headers["X-API-Key"] = key
        path, params = self._route(spec.name, parameters)
        logger.debug("user-data client request_id=%s tool=%s endpoint=%s", request_id, spec.name, path)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport, trust_env=False,
            ) as client:
                response = await client.get(path, params=params, headers=headers)
            if response.status_code >= 400:
                return self._from_api_error(spec, response, request_id, started)
            try:
                data = spec.output_model.model_validate(response.json())
            except (ValueError, ValidationError):
                logger.warning("user-data client invalid response request_id=%s tool=%s endpoint=%s", request_id, spec.name, path)
                return self._error(spec, request_id, started, "INTERNAL_ERROR", "Invalid user-data response", 500, False)
            empty = (spec.name == "get_margin_account" and not data.daily) or (
                spec.name == "get_portfolio_positions" and not data.stocks and not data.industries
            )
            return ToolResult[spec.output_model](
                status="empty" if empty else "success", data=data, source="synthetic_user_db",
                latency=_elapsed(started), error=None, request_id=request_id,
            )
        except httpx.TimeoutException:
            logger.warning("user-data client timeout request_id=%s tool=%s endpoint=%s", request_id, spec.name, path)
            return self._error(spec, request_id, started, "TIMEOUT", "User-data service request timed out", 504, True)
        except httpx.ConnectError:
            logger.warning("user-data client connection failure request_id=%s tool=%s endpoint=%s", request_id, spec.name, path)
            return self._error(spec, request_id, started, "DATA_UNAVAILABLE", "User-data service unavailable", 503, True)
        except httpx.HTTPError:
            logger.warning("user-data client HTTP failure request_id=%s tool=%s endpoint=%s", request_id, spec.name, path)
            return self._error(spec, request_id, started, "DATA_UNAVAILABLE", "User-data service unavailable", 503, True)

    def _route(self, name: str, parameters) -> tuple[str, dict[str, str | int]]:
        user_id = quote(parameters.user_id, safe="")
        if name == "get_customer_context":
            return f"/v1/customers/{user_id}/context", {}
        if name == "get_margin_account":
            values = parameters.model_dump(mode="json", exclude_none=True)
            values.pop("user_id")
            return f"/v1/customers/{user_id}/margin-account", values
        if name == "get_portfolio_positions":
            return f"/v1/customers/{user_id}/portfolio/positions", {}
        if name == "get_portfolio_analytics":
            return f"/v1/customers/{user_id}/portfolio/analytics", {}
        raise ToolFailure("UNKNOWN_TOOL", "Unknown tool", 404)

    def _from_api_error(self, spec: ToolSpec, response: httpx.Response, request_id: UUID, started: float) -> ToolResult:
        try:
            api_error = ApiError.model_validate(response.json())
            error = ToolError(
                code=api_error.code, message=api_error.message,
                http_status=api_error.http_status, retryable=api_error.retryable,
            )
        except (ValueError, ValidationError):
            logger.warning("user-data client invalid error response request_id=%s tool=%s", request_id, spec.name)
            error = ToolError(code="INTERNAL_ERROR", message="Invalid user-data error response", http_status=500, retryable=False)
        return ToolResult[spec.output_model](
            status="error", data=None, source="synthetic_user_db", latency=_elapsed(started),
            error=error, request_id=request_id,
        )

    @staticmethod
    def _error(spec: ToolSpec | None, request_id: UUID, started: float, code: str, message: str, status: int, retryable: bool) -> ToolResult:
        error = ToolError(code=code, message=message, http_status=status, retryable=retryable)
        result_type = ToolResult[spec.output_model] if spec else ToolResult
        return result_type(
            status="error", data=None, source="synthetic_user_db", latency=_elapsed(started),
            error=error, request_id=request_id,
        )


def _elapsed(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000)


def _run_sync(awaitable):
    """Run async client calls from the existing synchronous ToolRegistry boundary."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result = []
    failure = []

    def runner():
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:  # pragma: no cover - defensive loop bridge
            failure.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]
