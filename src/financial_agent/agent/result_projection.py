"""Safe public projection of Tool execution results for model prompts."""

from typing import Any

from financial_agent.agent.models import TaskExecutionResult


def project_result(item: TaskExecutionResult) -> dict[str, Any]:
    result = item.result
    return {
        "task_id": item.task_id,
        "tool_name": item.tool_name,
        "status": result.status,
        "data": result.data,
        "source": result.source,
        "error": result.error.model_dump(mode="json") if result.error else None,
        "retry_count": item.retry_count,
        "max_retry": item.max_retry,
    }
