"""Host-owned API-key mapping. No caller-supplied scopes are trusted."""

import hmac
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from financial_agent.tools.contracts import ToolFailure

Scope = Literal["read:profile", "read:portfolio"]


class Credential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    api_key: SecretStr = Field(exclude=True, repr=False, min_length=1)
    principal_id: str = Field(min_length=1, max_length=128)
    user_ids: frozenset[str]
    scopes: frozenset[Scope]


class CallContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)


class CredentialStore:
    def __init__(self, credentials: list[Credential]):
        keys = [item.api_key.get_secret_value() for item in credentials]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate API-key configuration")
        self._credentials = tuple(credentials)

    @classmethod
    def from_json(cls, value: SecretStr) -> "CredentialStore":
        try:
            records = json.loads(value.get_secret_value())
            if not isinstance(records, list):
                raise ValueError()
            return cls([Credential.model_validate(record) for record in records])
        except (ValueError, TypeError, ValidationError):
            raise ValueError("Invalid user API-key configuration") from None

    def authenticate(self, context: CallContext) -> Credential:
        supplied = context.api_key.get_secret_value() if context.api_key is not None else ""
        for credential in self._credentials:
            if hmac.compare_digest(supplied.encode("utf-8"), credential.api_key.get_secret_value().encode("utf-8")):
                return credential
        raise ToolFailure("UNAUTHORIZED", "Missing or invalid API key", 401)

    @staticmethod
    def require_scope(credential: Credential, scope: Scope) -> None:
        if scope not in credential.scopes:
            raise ToolFailure("FORBIDDEN", "Required scope is not granted", 403)

    @staticmethod
    def require_user(credential: Credential, user_id: str) -> None:
        if user_id not in credential.user_ids:
            raise ToolFailure("FORBIDDEN", "User access is not granted", 403)
