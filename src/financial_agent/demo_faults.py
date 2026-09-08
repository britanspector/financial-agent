"""Test/demo-only fault hook. Never wired by normal runtime assembly."""

from collections import deque
from collections.abc import Iterable
from typing import Literal

from financial_agent.tools.contracts import ToolFailure

Fault = Literal["timeout", "429", "503", "success"]


class FaultSequence:
    def __init__(self, outcomes: Iterable[Fault]):
        self._outcomes = deque(outcomes)
        if any(item not in {"timeout", "429", "503", "success"} for item in self._outcomes):
            raise ValueError("Unsupported demo fault")

    def __call__(self) -> None:
        outcome = self._outcomes.popleft() if self._outcomes else "success"
        if outcome == "timeout":
            raise ToolFailure("TIMEOUT", "Synthetic request timed out", 504, retryable=True)
        if outcome == "429":
            raise ToolFailure("RATE_LIMITED", "Synthetic rate limit reached", 429, retryable=True)
        if outcome == "503":
            raise ToolFailure("TEMPORARY_FAILURE", "Synthetic service temporarily unavailable", 503, retryable=True)
