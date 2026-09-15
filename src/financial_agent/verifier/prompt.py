"""Prompt and strict response schema for evidence-sufficiency verification."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.agent.result_projection import project_result
from financial_agent.schemas import UserQuery
from financial_agent.verifier.models import DraftAnswer
from financial_agent.context.models import HistorySummary, RetrievedHistoryTurn
from financial_agent.context.retrieval import retrieved_history_payload


VERIFICATION_RULES = (
    "Judge whether the current Tool results are sufficient to answer the user's request. "
    "First enumerate every fact or sub-question explicitly requested by the user, then locate direct support for each one in the "
    "current Tool results (not merely in the draft). If even one requested fact is absent, the decision must be REPLAN. "
    "Return PASS only when the draft answers the material request and its important claims are supported by the current results. "
    "Return REWRITE when the current Tool results already contain all necessary evidence but the draft omits it, contradicts it, "
    "cites it incorrectly, or needs clearer expression. Poor writing alone is never a reason to REPLAN. "
    "Return REPLAN only when the current Tool results themselves lack necessary evidence and another, different, or repeated Tool "
    "execution is required. A draft that admits a requested fact is unavailable does not answer that part of the request. "
    "REWRITE may reorganize or correct facts already present, but it may not infer, invent, omit, or retrieve an absent requested fact. "
    "Do not request new evidence that is already present. "
    "PASS and REWRITE must return missing_evidence as an empty list. REPLAN must identify the evidence that a new Tool execution "
    "must obtain. If the reason says evidence is absent or a new Tool execution is required, the decision must be REPLAN, never "
    "REWRITE. Before returning, verify that decision, reason, and missing_evidence express the same conclusion. "
    "failed_task_ids are supplied as deterministic facts; do not generate that field."
)


def verifier_response_schema() -> dict[str, Any]:
    def branch(decision: str, *, missing: bool) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {"const": decision},
                "reason": {"type": "string", "minLength": 1},
                "missing_evidence": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1 if missing else 0,
                    **({} if missing else {"maxItems": 0}),
                },
            },
            "required": ["decision", "reason", "missing_evidence"],
        }

    return {"oneOf": [
        branch("PASS", missing=False),
        branch("REWRITE", missing=False),
        branch("REPLAN", missing=True),
    ]}


def build_verifier_messages(
    request: UserQuery,
    plan: list[Task],
    tool_results: list[TaskExecutionResult],
    draft: DraftAnswer,
    *,
    resolved_evidence: list[dict[str, Any]],
    failed_task_ids: list[str],
    history_summary: HistorySummary | None = None,
    retrieved_history: list[RetrievedHistoryTurn] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "query": request.query,
        "history": [item.model_dump(mode="json") for item in request.history],
        "plan": [item.model_dump(mode="json") for item in plan],
        "tool_results": [project_result(item) for item in tool_results],
        "draft": {
            "answer": draft.answer,
            "evidence": resolved_evidence,
        },
        "failed_task_ids": failed_task_ids,
    }
    if history_summary is not None:
        payload["history_summary"] = history_summary.model_dump(mode="json")
    if retrieved_history:
        payload["retrieved_history"] = retrieved_history_payload(retrieved_history)
    json_payload = TypeAdapter(Any).dump_python(payload, mode="json")
    return [
        {
            "role": "system",
            "content": "You are a financial answer verifier. Return strict JSON matching the supplied schema. "
            + VERIFICATION_RULES
            + (" Treat history_summary as grounded stable history. Resolve conflicts using current query, raw recent "
               "history, retrieved raw history, then history_summary."
               if history_summary is not None or retrieved_history else ""),
        },
        {
            "role": "user",
            "content": json.dumps(json_payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
