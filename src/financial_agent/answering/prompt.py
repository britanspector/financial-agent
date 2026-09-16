"""Prompts and strict schema for evidence-grounded answer writing."""

import json
from typing import Any

from pydantic import TypeAdapter

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.agent.result_projection import project_result
from financial_agent.schemas import UserQuery
from financial_agent.verifier.models import DraftAnswer, VerificationResult
from financial_agent.context.models import HistorySummary, RetrievedHistoryTurn
from financial_agent.context.retrieval import retrieved_history_payload


def answer_response_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "source_path": {"type": "array", "minItems": 1,
                                    "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}},
                },
                "required": ["task_id", "source_path"],
            }},
        },
        "required": ["answer", "evidence"],
    }


def build_answer_messages(
    request: UserQuery,
    plan: list[Task],
    results: list[TaskExecutionResult],
    *,
    previous_draft: DraftAnswer | None = None,
    feedback: VerificationResult | None = None,
    history_summary: HistorySummary | None = None,
    retrieved_history: list[RetrievedHistoryTurn] | None = None,
) -> list[dict[str, str]]:
    rewriting = previous_draft is not None
    system = (
        "You are a financial answer writer. Return strict JSON matching the schema. Answer the user using only the supplied Tool "
        "results. Every material factual claim must cite an exact public value using task_id and a source_path rooted at "
        "ToolResult.data. source_path starts inside data, so cite a data field named value as [\"value\"], never "
        "[\"data\",\"value\"]. Do not cite ToolResult status, source, latency, error, request_id, or other envelope fields. "
        "Do not invent missing facts."
    )
    if rewriting:
        system += (
            " Rewrite the answer to address verifier feedback using only the same Tool results. Do not request, assume, or imply "
            "new Tool execution."
        )
    payload = {
        "query": request.query,
        "history": [item.model_dump(mode="json") for item in request.history],
        "plan": [item.model_dump(mode="json") for item in plan],
        "tool_results": [project_result(item) for item in results],
        "previous_draft": previous_draft.model_dump(mode="json") if previous_draft else None,
        "verifier_feedback": feedback.model_dump(mode="json") if feedback else None,
    }
    if history_summary is not None:
        system += " Treat history_summary as grounded stable history."
    if retrieved_history:
        payload["retrieved_history"] = retrieved_history_payload(retrieved_history)
    if history_summary is not None or retrieved_history:
        system += " Resolve conflicts using current query, raw recent history, retrieved raw history, then history_summary."
    serialized = TypeAdapter(Any).dump_python(payload, mode="json")
    messages = [{"role": "system", "content": system}]
    if history_summary is not None:
        messages.append({
            "role": "user",
            "content": json.dumps(
                {"history_summary": history_summary.model_dump(mode="json")},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        })
    messages.append({
        "role": "user", "content": json.dumps(serialized, ensure_ascii=False, separators=(",", ":")),
    })
    return messages
