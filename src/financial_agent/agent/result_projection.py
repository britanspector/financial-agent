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
        "data": public_result_data(result.data),
        "source": result.source,
        "error": result.error.model_dump(mode="json") if result.error else None,
        "retry_count": item.retry_count,
        "max_retry": item.max_retry,
    }


def public_result_data(value: Any) -> Any:
    """Project typed public Tool data into JSON-compatible prompt content."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [public_result_data(item) for item in value]
    if isinstance(value, dict):
        return {key: public_result_data(item) for key, item in value.items()}
    return value
