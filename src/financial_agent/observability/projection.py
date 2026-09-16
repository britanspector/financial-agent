"""The single projection and sanitization boundary for Agent traces."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, SecretStr

from financial_agent.agent.models import ResultBinding, Task
from financial_agent.agent.result_projection import public_result_data
from financial_agent.observability.models import (
    BindingProjection, BudgetProjection, EvidenceProjection, TaskProjection,
    TraceCaptureMode, ValueProjection,
)
from financial_agent.user_data.auth import CallContext
from financial_agent.verifier.models import EvidenceReference


_CREDENTIAL_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "password", "authorization",
    "credential", "credentials", "client_secret", "private_key",
}
_SAFE_RAW_BY_KIND = {
    "run_started": {"query", "history"},
    "context_selected": {"selected_content", "retrieved_content"},
    "plan_proposed": {"force_rerun_task_ids"},
    "tool_reuse": {"task_id"},
    "tool_attempt": {"task_id"},
    "tool_attempt_blocked": {"task_id"},
    "retry_scheduled": {"task_id"},
    "retry_skipped": {"task_id"},
    "writer_completed": {"answer"},
    "verifier_completed": {"failed_task_ids", "reason", "missing_evidence"},
    "run_finished": {"final_answer"},
}


def _is_forbidden_key(name: str) -> bool:
    folded = name.casefold()
    return (
        name.startswith("_") or folded in _CREDENTIAL_KEYS
        or folded.endswith(("_api_key", "_token", "_password", "_secret", "_credential"))
        or folded.startswith(("secret_", "credential_"))
    )


class TraceProjectionError(ValueError):
    pass


class TraceProjector:
    def __init__(self, capture_mode: TraceCaptureMode, run_key: bytes) -> None:
        self.capture_mode = capture_mode
        self._run_key = run_key

    def reference(self, value: Any) -> str:
        return hmac.new(self._run_key, self._canonical(value), hashlib.sha256).hexdigest()

    def event_fields(self, kind: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Final capture-mode-aware projection gate used by every recorder method."""
        if self.capture_mode != "safe":
            return fields
        for name in _SAFE_RAW_BY_KIND.get(kind, set()):
            if fields.get(name) is not None:
                raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
        self._reject_safe_nested(fields)
        return fields

    def value(self, value: Any, *, public: bool = True) -> ValueProjection:
        clean = self._public_json(value) if public else None
        types = sorted(self._value_types(clean))
        count = len(clean) if isinstance(clean, (dict, list, tuple)) else int(clean is not None)
        return ValueProjection(
            capture_mode=self.capture_mode,
            item_count=count,
            value_types=types,
            digest=self.reference(clean),
            value=clean if self.capture_mode == "evaluation" else None,
        )

    def task(self, task: Task) -> TaskProjection:
        task_ref = self.reference(task.task_id)
        bindings = [self.binding(item) for item in task.bindings]
        return TaskProjection(
            task_ref=task_ref,
            task_id=task.task_id if self.capture_mode == "evaluation" else None,
            tool_name=task.tool_name,
            arguments=self.value(task.arguments),
            dependency_refs=[self.reference(item) for item in task.dependencies],
            dependency_ids=list(task.dependencies) if self.capture_mode == "evaluation" else None,
            bindings=bindings,
        )

    def binding(self, binding: ResultBinding) -> BindingProjection:
        return BindingProjection(
            target_ref=self.reference(binding.target_parameter),
            source_task_ref=self.reference(binding.source_task_id),
            source_path_ref=self.reference(binding.source_path),
            target_parameter=binding.target_parameter if self.capture_mode == "evaluation" else None,
            source_task_id=binding.source_task_id if self.capture_mode == "evaluation" else None,
            source_path=list(binding.source_path) if self.capture_mode == "evaluation" else None,
        )

    def evidence(self, item: EvidenceReference) -> EvidenceProjection:
        return EvidenceProjection(
            task_ref=self.reference(item.task_id),
            path_ref=self.reference(item.source_path),
            task_id=item.task_id if self.capture_mode == "evaluation" else None,
            source_path=list(item.source_path) if self.capture_mode == "evaluation" else None,
        )

    def result(self, data: Any) -> ValueProjection:
        return self.value(public_result_data(data))

    def text(self, value: str) -> tuple[str, str | None]:
        return self.reference(value), value if self.capture_mode == "evaluation" else None

    def budget(self, budget: Any) -> BudgetProjection:
        return BudgetProjection(
            limit=budget.max_attempts,
            used=budget.attempt_count,
            remaining=budget.remaining,
            exhausted=budget.exhausted,
            deadline_exceeded=bool(getattr(budget, "deadline_exceeded", False)),
        )

    def _public_json(self, value: Any) -> Any:
        if isinstance(value, (CallContext, SecretStr)):
            raise TraceProjectionError("TRACE_FORBIDDEN_TYPE")
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (UUID, date, datetime, Decimal, Enum)):
            return str(value.value if isinstance(value, Enum) else value)
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                name = str(key)
                if _is_forbidden_key(name):
                    raise TraceProjectionError("TRACE_FORBIDDEN_KEY")
                result[name] = self._public_json(item)
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._public_json(item) for item in value]
        raise TraceProjectionError("TRACE_NON_JSON_VALUE")

    @staticmethod
    def _value_types(value: Any) -> set[str]:
        if value is None:
            return {"null"}
        if isinstance(value, bool):
            return {"boolean"}
        if isinstance(value, str):
            return {"string"}
        if isinstance(value, int):
            return {"integer"}
        if isinstance(value, float):
            return {"number"}
        if isinstance(value, list):
            nested = {kind for item in value for kind in TraceProjector._value_types(item)}
            return {"array", *nested}
        if isinstance(value, dict):
            nested = {kind for item in value.values() for kind in TraceProjector._value_types(item)}
            return {"object", *nested}
        return {type(value).__name__}

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()

    def _reject_safe_nested(self, value: Any) -> None:
        if isinstance(value, ValueProjection):
            if value.value is not None:
                raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
            return
        if isinstance(value, (TaskProjection, BindingProjection, EvidenceProjection)):
            for key in ("task_id", "dependency_ids", "target_parameter", "source_task_id", "source_path"):
                if hasattr(value, key) and getattr(value, key) is not None:
                    raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
            for key in type(value).model_fields:
                self._reject_safe_nested(getattr(value, key))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self._reject_safe_nested(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._reject_safe_nested(item)


class TraceSanitizer:
    """Defense-in-depth validation; evaluation mode is still not unrestricted."""

    def validate(self, value: Any, capture_mode: TraceCaptureMode) -> None:
        if capture_mode == "safe":
            self._reject_safe_raw(value)
        raw = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
        self._walk(raw, capture_mode)
        try:
            json.dumps(raw, ensure_ascii=False, allow_nan=False, default=self._json_default)
        except (TypeError, ValueError) as exc:
            raise TraceProjectionError("TRACE_SERIALIZATION_REJECTED") from exc

    def _walk(self, value: Any, capture_mode: TraceCaptureMode) -> None:
        if isinstance(value, (CallContext, SecretStr)):
            raise TraceProjectionError("TRACE_FORBIDDEN_TYPE")
        if isinstance(value, BaseModel):
            self._walk(value.model_dump(mode="python"), capture_mode)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                if _is_forbidden_key(name):
                    raise TraceProjectionError("TRACE_FORBIDDEN_KEY")
                self._walk(item, capture_mode)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._walk(item, capture_mode)
            return
        if value is None or isinstance(value, (str, int, float, bool, UUID)):
            return
        raise TraceProjectionError("TRACE_NON_JSON_VALUE")

    def _reject_safe_raw(self, value: Any) -> None:
        if isinstance(value, ValueProjection) and value.value is not None:
            raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
        if isinstance(value, (TaskProjection, BindingProjection, EvidenceProjection)):
            for key in ("task_id", "dependency_ids", "target_parameter", "source_task_id", "source_path"):
                if hasattr(value, key) and getattr(value, key) is not None:
                    raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
        if isinstance(value, BaseModel):
            kind = getattr(value, "kind", None)
            for key in _SAFE_RAW_BY_KIND.get(kind, set()):
                if getattr(value, key, None) is not None:
                    raise TraceProjectionError("TRACE_SAFE_RAW_VALUE")
            for key in type(value).model_fields:
                self._reject_safe_raw(getattr(value, key))
        elif isinstance(value, Mapping):
            for item in value.values():
                self._reject_safe_raw(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._reject_safe_raw(item)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        raise TypeError("not JSON serializable")
