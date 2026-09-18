"""Explicit local sinks for complete Agent traces."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Protocol

from financial_agent.observability.models import AgentTrace
from financial_agent.observability.projection import TraceSanitizer


class TraceSink(Protocol):
    def write(self, trace: AgentTrace) -> None: ...


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.traces: list[AgentTrace] = []
        self._lock = Lock()

    def write(self, trace: AgentTrace) -> None:
        TraceSanitizer().validate(trace, trace.capture_mode)
        with self._lock:
            self.traces.append(trace.model_copy(deep=True))


class JsonlTraceSink:
    """One validated, complete run per line; process-local append safety."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def write(self, trace: AgentTrace) -> None:
        TraceSanitizer().validate(trace, trace.capture_mode)
        line = trace.model_dump_json(exclude_none=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(line)
                stream.flush()


def load_agent_traces(path: str | Path) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw.get("schema_version") != "1.0":
                raise ValueError("unsupported schema version")
            trace = AgentTrace.model_validate(raw)
            TraceSanitizer().validate(trace, trace.capture_mode)
            traces.append(trace)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid Agent trace at line {number}") from exc
    return traces
