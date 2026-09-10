"""Authorization and result boundary around injected data/audit adapters."""

from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID, uuid4

from pydantic import ValidationError

from financial_agent.tools.contracts import ToolFailure, ToolResult
from financial_agent.tools.registry import ToolSpec
from financial_agent.user_data.audit import AuditEvent, AuditSink
from financial_agent.user_data.auth import CallContext, CredentialStore
from financial_agent.user_data.models import MarginAccount, PortfolioPositions
from financial_agent.user_data.repository import RepositoryUnavailable, UserDataRepository, UserNotFound


class UserDataService:
    def __init__(
        self, repository: UserDataRepository, credentials: CredentialStore, audit: AuditSink,
        *, clock: Callable[[], float] = perf_counter,
        before_read: Callable[[], None] | None = None,
    ):
        self._repository = repository
        self._credentials = credentials
        self._audit = audit
        self._clock = clock
        self._before_read = before_read

    def execute(
        self, spec: ToolSpec | None, arguments: object, *, context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult:
        started = self._clock()
        request_id = request_id or uuid4()
        principal_id = user_id = None
        data = error = None
        status = "error"
        try:
            credential = self._credentials.authenticate(context)
            principal_id = credential.principal_id
            if spec is None:
                raise ToolFailure("UNKNOWN_TOOL", "Unknown tool", 404)
            self._credentials.require_scope(credential, spec.scope)
            try:
                parameters = spec.input_model.model_validate(arguments)
            except ValidationError:
                raise ToolFailure("INVALID_ARGUMENT", "Invalid tool arguments", 422) from None
            user_id = parameters.user_id
            self._credentials.require_user(credential, user_id)
            if self._before_read is not None:
                self._before_read()
            operations = {
                "customer_context": lambda: self._repository.get_customer_context(user_id),
                "margin_account": lambda: self._repository.get_margin_account(parameters),
                "portfolio_positions": lambda: self._repository.get_portfolio_positions(user_id),
                "portfolio_analytics": lambda: self._repository.get_portfolio_analytics(user_id),
            }
            if spec.operation not in operations:
                raise RuntimeError("Unsupported internal operation")
            data = operations[spec.operation]()
            data = spec.output_model.model_validate(data)
            empty = (
                isinstance(data, MarginAccount) and not data.daily
            ) or (
                isinstance(data, PortfolioPositions) and not data.stocks and not data.industries
            )
            status = "empty" if empty else "success"
        except ToolFailure as exc:
            error = exc.error
        except UserNotFound:
            error = ToolFailure("NOT_FOUND", "Synthetic user not found", 404).error
        except RepositoryUnavailable:
            error = ToolFailure("DATA_UNAVAILABLE", "Synthetic data store unavailable", 503, retryable=True).error
        except Exception:
            # Never expose SQL, local paths, validation input, or adapter exception text.
            error = ToolFailure("INTERNAL_ERROR", "Internal tool failure", 500).error
        elapsed = max(0.0, (self._clock() - started) * 1000)
        if error is not None:
            data, status = None, "error"
        try:
            self._audit.write(AuditEvent(
                timestamp=datetime.now(timezone.utc), request_id=request_id,
                tool=spec.name if spec else "unknown", principal_id=principal_id, user_id=user_id,
                status=status, code=error.code if error else status.upper(),
                http_status=error.http_status if error else 200, latency=elapsed,
            ))
        except Exception:
            data, status = None, "error"
            error = ToolFailure("AUDIT_UNAVAILABLE", "Audit write failed", 500).error
        result_type = ToolResult[spec.output_model] if spec else ToolResult
        return result_type(
            status=status, data=data, source="synthetic_user_db", latency=elapsed,
            error=error, request_id=request_id,
        )
