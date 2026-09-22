"""Local-only developer console API for traced Agent runs.

The console deliberately keeps provider credentials and the process-local user
tool credential behind this server boundary. It is intended to be bound to
127.0.0.1, not exposed as a production API.
"""

from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import Field, SecretStr

from financial_agent.agent.loop import AgentLoopResult, LoopPolicy
from financial_agent.agent.retry import RetryPolicy
from financial_agent.answering.runtime import build_answer_writer
from financial_agent.config import Settings
from financial_agent.knowledge.runtime import build_rag_tools
from financial_agent.market_data.runtime import build_market_tools
from financial_agent.observability.models import AgentTrace, TraceCaptureMode
from financial_agent.observability.recorder import AgentTraceRecorder
from financial_agent.observability.runtime import run_agent_loop_traced
from financial_agent.planner.runtime import build_planner
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.composite import CompositeToolRegistry, merge_registries
from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CallContext, Credential, CredentialStore
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.service import UserDataService
from financial_agent.verifier.runtime import build_verifier


class DevRunRequest(Schema):
    query: str = Field(min_length=1, max_length=20_000)
    history: list[Message] = Field(default_factory=list, max_length=200)
    capture_mode: TraceCaptureMode = "evaluation"


class DevRunStarted(Schema):
    run_id: UUID
    request_id: UUID
    status: Literal["queued"] = "queued"


@dataclass
class _RunJob:
    run_id: UUID
    request: UserQuery
    recorder: AgentTraceRecorder
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    result: AgentLoopResult | None = None
    trace: AgentTrace | None = None
    error_code: str | None = None
    error_type: str | None = None
    diagnostics: dict[str, Any] | None = None
    lock: Lock = field(default_factory=Lock)


class DevAgentRuntime:
    """One process-local runtime; jobs are serialized to keep providers isolated."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry, self.call_context, self.rag_enabled = _build_registry(settings)
        self.planner, self.validator = build_planner(settings, self.registry)
        self.writer = build_answer_writer(settings, self.registry)
        self.verifier = build_verifier(settings, self.registry)
        self.loop_policy = LoopPolicy.from_settings(settings)
        self.retry_policy = RetryPolicy.from_settings(settings)

    def run(self, job: _RunJob) -> None:
        traced = run_agent_loop_traced(
            job.request,
            self.planner,
            self.validator,
            self.registry,
            self.writer,
            self.verifier,
            policy=self.loop_policy,
            retry_policy=self.retry_policy,
            context=self.call_context,
            max_concurrency=4,
            capture_mode=job.recorder.capture_mode,
            recorder=job.recorder,
        )
        with job.lock:
            job.result = traced.result
            job.trace = traced.trace

    def failure_diagnostics(self) -> dict[str, Any] | None:
        """Expose only structured evidence paths in local evaluation mode."""
        provider = getattr(self.writer, "_provider", None)
        raw = getattr(provider, "last_raw_response", None)
        if not isinstance(raw, dict):
            return None
        evidence = raw.get("evidence")
        return {"answer_evidence": evidence} if isinstance(evidence, list) else None


class DevRunStore:
    def __init__(self, runtime: DevAgentRuntime) -> None:
        self.runtime = runtime
        self._jobs: dict[UUID, _RunJob] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="financial-agent-dev")

    def start(self, payload: DevRunRequest) -> _RunJob:
        request = UserQuery(query=payload.query, history=payload.history)
        job = _RunJob(
            run_id=uuid4(),
            request=request,
            recorder=AgentTraceRecorder(request.request_id, capture_mode=payload.capture_mode),
        )
        with self._lock:
            if len(self._jobs) >= 50:
                oldest = next(iter(self._jobs))
                self._jobs.pop(oldest)
            self._jobs[job.run_id] = job
        self._executor.submit(self._execute, job)
        return job

    def get(self, run_id: UUID) -> _RunJob | None:
        with self._lock:
            return self._jobs.get(run_id)

    def _execute(self, job: _RunJob) -> None:
        with job.lock:
            job.status = "running"
        try:
            self.runtime.run(job)
        except BaseException as exc:
            # Provider exception messages can contain response material. Only
            # expose a stable code and the exception class to the local UI.
            with job.lock:
                job.error_code = "AGENT_RUN_FAILED"
                job.error_type = type(exc).__name__
                diagnostic = getattr(self.runtime, "failure_diagnostics", None)
                job.diagnostics = diagnostic() if callable(diagnostic) else None
                job.status = "failed"
        else:
            with job.lock:
                job.status = "completed"

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _build_registry(settings: Settings) -> tuple[CompositeToolRegistry, CallContext, bool]:
    if settings.qwen_api_key is None:
        raise ValueError("Qwen API key is required for the developer console")
    if not settings.user_db_path.exists():
        raise ValueError("Synthetic user database is missing; run generate-synthetic-data")

    local_key = SecretStr("dev-local-" + secrets.token_urlsafe(24))
    credential = Credential(
        api_key=local_key,
        principal_id="financial-agent-dev-console",
        user_ids=frozenset(f"syn-user-{index:04d}" for index in range(1, 2001)),
        scopes=frozenset({
            "read:customer_context",
            "read:margin_account",
            "read:portfolio_positions",
            "read:portfolio_analytics",
        }),
    )
    user_service = UserDataService(
        repository=SQLiteUserDataRepository(settings.user_db_path),
        credentials=CredentialStore([credential]),
        audit=JsonlAuditSink(Path("data/dev-console-audit.jsonl")),
    )
    registries = [register_user_tools(user_service)]
    if settings.tushare_token is not None:
        registries.append(build_market_tools(settings))
    rag_enabled = settings.rag_embedding_index_path.exists()
    if rag_enabled:
        registries.append(build_rag_tools(settings))
    return merge_registries(*registries), CallContext(api_key=local_key), rag_enabled


def _job_snapshot(job: _RunJob) -> dict[str, Any]:
    with job.lock:
        status = job.status
        result = job.result
        trace = job.trace
        error_code = job.error_code
        error_type = job.error_type
        diagnostics = job.diagnostics
    events = trace.events if trace is not None else job.recorder.events
    return {
        "run_id": str(job.run_id),
        "request_id": str(job.request.request_id),
        "trace_id": str(job.recorder.trace_id),
        "status": status,
        "capture_mode": job.recorder.capture_mode,
        "events": [event.model_dump(mode="json", exclude_none=True) for event in events],
        "result": result.model_dump(mode="json", exclude_none=True) if result else None,
        "trace": trace.model_dump(mode="json", exclude_none=True) if trace else None,
        "error": (
            {"code": error_code, "type": error_type}
            if error_code is not None else None
        ),
        "diagnostics": diagnostics if job.recorder.capture_mode == "evaluation" else None,
    }


def create_dev_console_app(
    settings: Settings | None = None,
    *,
    runtime: DevAgentRuntime | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    settings = settings or Settings()
    runtime = runtime or DevAgentRuntime(settings)
    store = DevRunStore(runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(title="Financial Agent Developer Console", version="1.0", lifespan=lifespan)

    @app.get("/api/environment")
    async def environment() -> dict[str, Any]:
        return {
            "environment": "local",
            "qwen_configured": settings.qwen_api_key is not None,
            "tushare_configured": settings.tushare_token is not None,
            "rag_enabled": runtime.rag_enabled,
            "capture_modes": ["safe", "evaluation"],
            "models": {
                "planner": settings.planner_model,
                "writer": settings.answer_model,
                "verifier": settings.verifier_model,
            },
            "tools": [item["name"] for item in runtime.registry.describe()],
        }

    @app.post("/api/runs", response_model=DevRunStarted, status_code=202)
    async def start_run(payload: DevRunRequest) -> DevRunStarted:
        job = store.start(payload)
        return DevRunStarted(run_id=job.run_id, request_id=job.request.request_id)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: UUID) -> dict[str, Any]:
        job = store.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _job_snapshot(job)

    web_root = frontend_dir or Path(__file__).resolve().parents[2] / "frontend"
    if not web_root.is_dir():
        raise ValueError("Developer console frontend directory is missing")
    app.mount("/", StaticFiles(directory=web_root, html=True), name="frontend")
    return app
