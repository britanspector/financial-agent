"""FastAPI transport for the local synthetic user-data service."""

from datetime import date
from logging import getLogger
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from financial_agent.config import Settings
from financial_agent.tools.registry import ToolRegistry
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.models import (
    CustomerContext, MarginAccount, PortfolioAnalytics, PortfolioPositions,
)
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.service import UserDataService
from financial_agent.user_data.http_models import ApiError

logger = getLogger(__name__)


def _request_id(value: str | None) -> UUID:
    return UUID(value) if value else uuid4()


def _api_error(code: str, message: str, status: int, request_id: UUID, *, retryable: bool = False) -> JSONResponse:
    body = ApiError(
        code=code, message=message, http_status=status,
        retryable=retryable, request_id=request_id,
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        headers={"X-Request-ID": str(request_id)},
    )


def _reject_unknown_query_parameters(
    request: Request, allowed: frozenset[str], request_id: UUID,
) -> JSONResponse | None:
    if set(request.query_params) <= allowed:
        return None
    logger.info("user-data request validation failed request_id=%s", request_id)
    return _api_error("INVALID_ARGUMENT", "Invalid request parameters", 422, request_id)


def create_app(settings: Settings | None = None, *, service: UserDataService | None = None) -> FastAPI:
    """Build the HTTP app; business composition remains in the existing runtime."""
    if service is None:
        if settings is None:
            settings = Settings()
        from financial_agent.user_data.audit import JsonlAuditSink
        from financial_agent.user_data.auth import CredentialStore
        from financial_agent.user_data.repository import SQLiteUserDataRepository

        service = UserDataService(
            repository=SQLiteUserDataRepository(settings.user_db_path),
            credentials=CredentialStore.from_json(settings.user_api_keys),
            audit=JsonlAuditSink(settings.audit_path),
        )
    registry = register_user_tools(service)
    app = FastAPI(title="Synthetic User Data Service", version="1.0")

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        try:
            request_id = _request_id(request.headers.get("X-Request-ID"))
        except ValueError:
            request_id = uuid4()
        logger.info("user-data request validation failed request_id=%s", request_id)
        return _api_error("INVALID_ARGUMENT", "Invalid request parameters", 422, request_id)

    async def invoke(
        name: str,
        arguments: dict,
        output_model: type,
        api_key: str | None,
        request_id: UUID,
    ) -> JSONResponse:
        result = registry.invoke(
            name, arguments, context=CallContext(api_key=api_key), request_id=request_id,
        )
        if result.status == "error":
            error = result.error
            assert error is not None
            return _api_error(
                error.code, error.message, error.http_status, result.request_id,
                retryable=error.retryable,
            )
        data = output_model.model_validate(result.data)
        return JSONResponse(
            status_code=200,
            content=data.model_dump(mode="json"),
            headers={"X-Request-ID": str(result.request_id)},
        )

    @app.get("/v1/customers/{user_id}/context", response_model=CustomerContext)
    async def customer_context(request: Request, user_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-ID")):
        try:
            request_id = _request_id(x_request_id)
        except ValueError:
            return _api_error("INVALID_ARGUMENT", "Invalid X-Request-ID", 422, uuid4())
        if error := _reject_unknown_query_parameters(request, frozenset(), request_id):
            return error
        return await invoke("get_customer_context", {"user_id": user_id}, CustomerContext, x_api_key, request_id)

    @app.get("/v1/customers/{user_id}/margin-account", response_model=MarginAccount)
    async def margin_account(
        request: Request,
        user_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = Query(default=60, ge=1, le=366),
        offset: int = Query(default=0, ge=0),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ):
        try:
            request_id = _request_id(x_request_id)
        except ValueError:
            return _api_error("INVALID_ARGUMENT", "Invalid X-Request-ID", 422, uuid4())
        if error := _reject_unknown_query_parameters(
            request, frozenset({"start_date", "end_date", "limit", "offset"}), request_id,
        ):
            return error
        arguments = {
            "user_id": user_id, "start_date": start_date, "end_date": end_date,
            "limit": limit, "offset": offset,
        }
        return await invoke("get_margin_account", arguments, MarginAccount, x_api_key, request_id)

    @app.get("/v1/customers/{user_id}/portfolio/positions", response_model=PortfolioPositions)
    async def portfolio_positions(request: Request, user_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-ID")):
        try:
            request_id = _request_id(x_request_id)
        except ValueError:
            return _api_error("INVALID_ARGUMENT", "Invalid X-Request-ID", 422, uuid4())
        if error := _reject_unknown_query_parameters(request, frozenset(), request_id):
            return error
        return await invoke("get_portfolio_positions", {"user_id": user_id}, PortfolioPositions, x_api_key, request_id)

    @app.get("/v1/customers/{user_id}/portfolio/analytics", response_model=PortfolioAnalytics)
    async def portfolio_analytics(request: Request, user_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-ID")):
        try:
            request_id = _request_id(x_request_id)
        except ValueError:
            return _api_error("INVALID_ARGUMENT", "Invalid X-Request-ID", 422, uuid4())
        if error := _reject_unknown_query_parameters(request, frozenset(), request_id):
            return error
        return await invoke("get_portfolio_analytics", {"user_id": user_id}, PortfolioAnalytics, x_api_key, request_id)

    return app
