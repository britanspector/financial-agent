"""Shared safe paths below public ``ToolResult.data`` values."""

from __future__ import annotations

from collections.abc import Mapping
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, StrictInt, StrictStr

ResultPathSegment = StrictStr | StrictInt


class ResultPathError(ValueError):
    """Raised when a public result path cannot be resolved safely."""


def result_path_type(model: type[BaseModel] | None, path: list[str | int]) -> Any | None:
    """Return the public type addressed by a field/list-index path."""
    current: Any = model
    for part in path:
        current = _unwrap_optional(current)
        current = _unwrap_root_model(current)
        origin = get_origin(current)
        if isinstance(current, type) and issubclass(current, BaseModel):
            if not isinstance(part, str) or part not in current.model_fields:
                return None
            current = current.model_fields[part].annotation
        elif origin is list:
            if not isinstance(part, int) or isinstance(part, bool) or part < 0:
                return None
            current = get_args(current)[0]
        else:
            return None
    return current


def resolve_result_path(data: Any, path: list[str | int]) -> Any:
    """Resolve the same safe path at runtime, starting strictly below data."""
    current = data
    try:
        for part in path:
            if isinstance(part, int) and not isinstance(part, bool):
                if part < 0 or not isinstance(current, list):
                    raise ResultPathError("Result path requires a non-negative list index")
                current = current[part]
            elif isinstance(part, str):
                if isinstance(current, Mapping):
                    current = current[part]
                elif isinstance(current, BaseModel) and part in current.__class__.model_fields:
                    current = getattr(current, part)
                else:
                    raise ResultPathError("Result path requires a public field name")
            else:
                raise ResultPathError("Result path segments must be field names or list indexes")
    except (IndexError, KeyError) as exc:
        raise ResultPathError("Result path value is unavailable") from exc
    return current


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        options = [item for item in get_args(annotation) if item is not type(None)]
        if len(options) == 1:
            return options[0]
    return annotation


def _unwrap_root_model(annotation: Any) -> Any:
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and getattr(annotation, "__pydantic_root_model__", False)
    ):
        return annotation.model_fields["root"].annotation
    return annotation
