"""Model-independent token estimation abstractions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from math import ceil
from typing import Protocol

from financial_agent.schemas import Message


class TokenEstimator(Protocol):
    def estimate_messages(self, messages: Sequence[Message]) -> int: ...


class HeuristicTokenEstimator:
    """Estimate tokens from stable compact-JSON UTF-8 size.

    This is deliberately a budget heuristic, not a claim about any model's
    tokenizer. A provider-specific estimator can be injected behind the same
    protocol later without changing context selection.
    """

    def estimate_messages(self, messages: Sequence[Message]) -> int:
        if not messages:
            return 0
        payload = [message.model_dump(mode="json") for message in messages]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return max(1, ceil(len(serialized.encode("utf-8")) / 4))
