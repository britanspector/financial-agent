"""Strict prompt contract for grounded extractive history summaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from financial_agent.schemas import Message


SUMMARY_SCHEMA_VERSION = "phase5.3-stable-v1"


def summary_response_schema(max_facts: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": max_facts,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {"enum": [
                            "entity", "time_range", "constraint", "confirmed_intent", "planning_fact",
                        ]},
                        "content": {"type": "string", "minLength": 1},
                        "source_message_index": {"type": "integer", "minimum": 0},
                    },
                    "required": ["category", "content", "source_message_index"],
                },
            },
        },
        "required": ["facts"],
    }


def build_summary_messages(
    summarized_messages: Sequence[tuple[int, Message]],
) -> list[dict[str, str]]:
    system = (
        "You extract grounded facts from old conversation history. Return strict JSON. "
        "Every fact content must be an exact non-empty substring of its indexed source message. "
        "Preserve user IDs, stock symbols, and dates byte-for-byte. Keep user constraints, corrected final values, "
        "confirmed intent, and facts needed for later planning; omit greetings, repetition, and ordinary assistant "
        "acknowledgements. A later correction replaces an older value and must cite the correction message. "
        "Summarize durable facts from summarized_history without adapting them to any current query."
    )
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summarized_history": [
            {"index": index, **message.model_dump(mode="json")}
            for index, message in summarized_messages
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
