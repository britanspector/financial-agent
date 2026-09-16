"""Safe public projection of Tool execution results for model prompts."""

from typing import Any

from pydantic import BaseModel

from financial_agent.agent.models import TaskExecutionResult


def project_result(item: TaskExecutionResult) -> dict[str, Any]:
    result = item.result
    return {
        "task_id": item.task_id,
        "tool_name": item.tool_name,
        "status": result.status,
        "data": _json_value(result.data),
        "source": result.source,
        "error": result.error.model_dump(mode="json") if result.error else None,
        "retry_count": item.retry_count,
        "max_retry": item.max_retry,
    }


def _json_value(value: Any) -> Any:
    """Project typed public Tool data into JSON-compatible prompt content."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value
