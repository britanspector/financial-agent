"""Run-scoped retry policy and execution-attempt budget."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING

from pydantic import Field

from financial_agent.schemas import Schema

if TYPE_CHECKING:
    from financial_agent.config import Settings


class RetryPolicy(Schema):
    """Deterministic retry and graph-wide attempt limits."""

    max_retry: int = Field(default=2, ge=0, le=100)
    initial_backoff_seconds: float = Field(default=0.5, ge=0, le=300)
    backoff_multiplier: float = Field(default=2.0, ge=1, le=100)
    max_attempts: int = Field(default=36, ge=1, le=10_000)
    deadline_seconds: float = Field(default=120.0, gt=0, le=86_400)

    def backoff_seconds(self, retry_count: int) -> float:
        """Return the delay before the next retry (zero-based retry index)."""
        return self.initial_backoff_seconds * (self.backoff_multiplier ** retry_count)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RetryPolicy":
        return cls(
            max_retry=settings.execution_max_retry,
            initial_backoff_seconds=settings.execution_initial_backoff_seconds,
            backoff_multiplier=settings.execution_backoff_multiplier,
            max_attempts=settings.execution_max_attempts,
            deadline_seconds=settings.execution_deadline_seconds,
        )


class ToolAttemptBudget:
    """Thread-safe Tool-attempt budget that may span multiple graph runs."""

    def __init__(self, max_attempts: int) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._max_attempts = max_attempts
        self._attempt_count = 0
        self._lock = Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self._attempt_count >= self._max_attempts:
                return False
            self._attempt_count += 1
            return True

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._attempt_count >= self._max_attempts

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._max_attempts - self._attempt_count)


class ExecutionBudget:
    """A graph-run deadline backed by an optionally loop-wide attempt budget."""

    def __init__(
        self,
        policy: RetryPolicy,
        *,
        clock: Callable[[], float],
        attempt_budget: ToolAttemptBudget | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._started_at = clock()
        self._attempt_budget = attempt_budget or ToolAttemptBudget(policy.max_attempts)
        self._attempt_budget_exhausted = False
        self._deadline_exceeded = False
        self._lock = Lock()

    def reserve_attempt(self) -> bool:
        """Reserve one attempt if it may start now.

        The deadline is deliberately not a cancellation timeout. Once this
        method succeeds, the synchronous Tool attempt may run past it.
        """
        with self._lock:
            if self._clock() >= self._started_at + self._policy.deadline_seconds:
                self._deadline_exceeded = True
                return False
            if not self._attempt_budget.reserve():
                self._attempt_budget_exhausted = True
                return False
            return True

    def allows_retry_after(self, delay_seconds: float) -> bool:
        """Check whether waiting can possibly lead to another attempt."""
        with self._lock:
            if self._attempt_budget.exhausted:
                self._attempt_budget_exhausted = True
                return False
            if self._clock() + delay_seconds >= self._started_at + self._policy.deadline_seconds:
                self._deadline_exceeded = True
                return False
            return True

    @property
    def attempt_count(self) -> int:
        return self._attempt_budget.attempt_count

    @property
    def attempt_budget_exhausted(self) -> bool:
        with self._lock:
            return self._attempt_budget_exhausted

    @property
    def deadline_exceeded(self) -> bool:
        with self._lock:
            return self._deadline_exceeded

    def elapsed_ms(self) -> float:
        return max(0.0, (self._clock() - self._started_at) * 1000)
