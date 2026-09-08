"""Separate JSONL audit sink; contains neither credentials nor result bodies."""

from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field

from financial_agent.schemas import Schema


class AuditEvent(Schema):
    timestamp: AwareDatetime
    request_id: UUID
    tool: str
    principal_id: str | None
    user_id: str | None
    status: Literal["success", "empty", "error"]
    code: str
    http_status: int
    latency: float = Field(ge=0)


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...


class JsonlAuditSink:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, event: AuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
