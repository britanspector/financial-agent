"""HTTP transport-only contracts; Agent ToolResult stays outside this layer."""

from uuid import UUID

from pydantic import Field

from financial_agent.schemas import Schema


class ApiError(Schema):
    code: str
    message: str
    http_status: int = Field(ge=400, le=599)
    retryable: bool
    request_id: UUID

