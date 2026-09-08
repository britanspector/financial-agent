"""Stable, JSON-serializable tool envelope."""

from typing import Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, model_validator

from financial_agent.schemas import Schema

T = TypeVar("T")


class ToolError(Schema):
    code: str
    message: str
    http_status: int = Field(ge=400, le=599)
    retryable: bool


class ToolResult(Schema, Generic[T]):
    status: Literal["success", "empty", "error"]
    data: T | None
    source: str
    latency: float = Field(ge=0, allow_inf_nan=False, description="Elapsed milliseconds")
    error: ToolError | None
    request_id: UUID

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.status == "error":
            if self.error is None or self.data is not None:
                raise ValueError("Errors require error details and null data")
        elif self.error is not None or self.data is None:
            raise ValueError("Successful and empty results require data and null error")
        return self


class ToolFailure(Exception):
    def __init__(self, code: str, message: str, http_status: int, *, retryable: bool = False):
        super().__init__(message)
        self.error = ToolError(code=code, message=message, http_status=http_status, retryable=retryable)
