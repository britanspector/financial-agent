"""Minimal request/state contracts, without planner or tool execution schemas."""

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Message(Schema):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class UserQuery(Schema):
    request_id: UUID = Field(default_factory=uuid4)
    query: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)


class AgentState(Schema):
    request: UserQuery
    status: Literal["pending", "completed", "failed"] = "pending"
