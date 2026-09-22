from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import Field, ValidationError

from financial_agent.agent.loop import LoopPolicy, run_agent_loop
from financial_agent.agent.e2e_evaluation import evaluate_agent_e2e, load_e2e_eval_set
from financial_agent.agent.models import Task
from financial_agent.agent.retry import RetryPolicy
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.observability import (
    AgentTrace, InMemoryTraceSink, JsonlTraceSink, TraceProjector, TraceSanitizer,
    load_agent_traces, run_agent_loop_traced,
)
from financial_agent.observability.models import ToolAttemptEvent
from financial_agent.observability.recorder import AgentTraceRecorder
from financial_agent.planner.models import StructuredPlan
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.contracts import ToolResult
from financial_agent.user_data.auth import CallContext
from financial_agent.verifier.models import DraftAnswer, EvidenceReference, VerificationResult


class _Input(Schema):
    user_id: str
    symbol: str


class _Output(Schema):
    fact: str


class _Registry:
    def describe(self):
        return [{"name": "lookup", "input_schema": _Input.model_json_schema(),
                 "output_schema": _Output.model_json_schema()}]

    def input_model(self, name):
        return _Input if name == "lookup" else None

    def output_model(self, name):
        return _Output if name == "lookup" else None

    def invoke(self, name, arguments, *, context, request_id=None):
        assert isinstance(context, CallContext)
        return ToolResult(
            status="success", data=_Output(fact="PRIVATE-RESULT"), source="fixture", latency=0,
            error=None, request_id=request_id or uuid4(),
        )


class _Planner:
    def plan(self, request):
        return StructuredPlan(decision="execute", tasks=[{
            "task_id": "private-task", "tool_name": "lookup",
            "arguments": {"user_id": "syn-user-secret", "symbol": "SECRET.SYMBOL"},
        }])


class _Writer:
    def write(self, request, plan, results):
        return DraftAnswer(
            answer="PRIVATE-ANSWER",
            evidence=[EvidenceReference(task_id="private-task", source_path=["fact"])],
        )


class _Verifier:
    def verify(self, request, plan, results, draft):
        return VerificationResult(decision="PASS", reason="PRIVATE-REASON")


def _run(*, mode="safe", sink=None, recorder=None):
    registry = _Registry()
    request = UserQuery(
        query="PRIVATE-QUERY", history=[Message(role="user", content="PRIVATE-HISTORY")],
    )
    return run_agent_loop_traced(
        request, _Planner(), PlanValidator(registry), registry, _Writer(), _Verifier(),
        policy=LoopPolicy(), retry_policy=RetryPolicy(initial_backoff_seconds=0),
        context=CallContext(api_key="PRIVATE-CREDENTIAL"), capture_mode=mode, sink=sink,
        recorder=recorder,
    )


def test_safe_trace_hides_content_arguments_and_result_values():
    traced = _run()
    serialized = traced.trace.model_dump_json()
    for sentinel in (
        "PRIVATE-QUERY", "PRIVATE-HISTORY", "syn-user-secret", "SECRET.SYMBOL",
        "PRIVATE-RESULT", "PRIVATE-ANSWER", "PRIVATE-REASON", "PRIVATE-CREDENTIAL",
        "user_id", "symbol",
    ):
        assert sentinel not in serialized
    attempt = next(event for event in traced.trace.events if isinstance(event, ToolAttemptEvent))
    assert attempt.arguments.value is None
    assert attempt.arguments.item_count == 2


def test_evaluation_trace_keeps_public_synthetic_values_but_not_call_context():
    traced = _run(mode="evaluation")
    serialized = traced.trace.model_dump_json(exclude_none=True)
    assert "PRIVATE-QUERY" in serialized
    assert "syn-user-secret" in serialized
    assert "SECRET.SYMBOL" in serialized
    assert "PRIVATE-RESULT" in serialized
    assert "PRIVATE-CREDENTIAL" not in serialized


def test_model_call_trace_keeps_exact_io_only_in_evaluation_mode():
    messages = [{"role": "system", "content": "PRIVATE-MODEL-CONTEXT"}]
    response_schema = {"type": "object", "required": ["decision"]}
    response = {"decision": "PASS", "reason": "PRIVATE-MODEL-RESPONSE"}

    evaluation = AgentTraceRecorder(uuid4(), capture_mode="evaluation")
    with evaluation.activate(), evaluation.scope(operation="verify", iteration=2, plan_revision=1):
        call_index = evaluation.record_model_call_started(
            "verifier", "fixture-model", messages, response_schema,
        )
        evaluation.record_model_call_completed(
            "verifier", call_index, "fixture-model", response,
        )

    started, completed = evaluation.events
    assert started.kind == "model_call_started"
    assert started.call_index == completed.call_index == 1
    assert started.messages.value == messages
    assert started.response_schema.value == response_schema
    assert completed.response.value == response
    assert started.iteration == completed.iteration == 2

    safe = AgentTraceRecorder(uuid4(), capture_mode="safe")
    with safe.activate():
        safe_index = safe.record_model_call_started(
            "verifier", "fixture-model", messages, response_schema,
        )
        safe.record_model_call_completed("verifier", safe_index, "fixture-model", response)
    serialized = json.dumps(
        [event.model_dump(mode="json", exclude_none=True) for event in safe.events],
    )
    assert "PRIVATE-MODEL-CONTEXT" not in serialized
    assert "PRIVATE-MODEL-RESPONSE" not in serialized
    assert all(
        projection.value is None
        for event in safe.events
        for projection in (
            [event.messages, event.response_schema]
            if event.kind == "model_call_started" else [event.response]
        )
    )


def test_plain_and_traced_loop_results_have_same_behavior():
    registry = _Registry()
    request = UserQuery(query="same")
    kwargs = dict(
        request=request, planner=_Planner(), validator=PlanValidator(registry), registry=registry,
        writer=_Writer(), verifier=_Verifier(), context=CallContext(),
        retry_policy=RetryPolicy(initial_backoff_seconds=0),
    )
    plain = run_agent_loop(**kwargs)
    traced = run_agent_loop_traced(**kwargs).result
    assert plain.model_dump(exclude={"task_results"}) == traced.model_dump(exclude={"task_results"})
    assert [(x.task_id, x.result.status, x.result.data) for x in plain.task_results] == [
        (x.task_id, x.result.status, x.result.data) for x in traced.task_results
    ]


def test_jsonl_round_trip_and_unknown_event_rejection(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    sink = JsonlTraceSink(path)
    first = _run(mode="evaluation", sink=sink)
    second = _run(mode="evaluation", sink=sink)
    assert first.persistence.status == second.persistence.status == "saved"
    loaded = load_agent_traces(path)
    assert [item.trace_id for item in loaded] == [first.trace.trace_id, second.trace.trace_id]
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    raw["events"][0]["kind"] = "unknown"
    with pytest.raises(ValidationError):
        AgentTrace.model_validate(raw)
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    raw["schema_version"] = "2.0"
    invalid = tmp_path / "invalid-version.jsonl"
    invalid.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Agent trace"):
        load_agent_traces(invalid)


def test_jsonl_sink_appends_complete_lines_from_threads(tmp_path: Path):
    path = tmp_path / "threads.jsonl"
    sink = JsonlTraceSink(path)
    trace = _run(mode="evaluation").trace
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: sink.write(trace), range(12)))
    assert len(load_agent_traces(path)) == 12


def test_sink_and_sanitizer_failures_do_not_change_success_result():
    class BadSink:
        def write(self, trace):
            raise OSError("sink detail must not escape")

    class BadSanitizer(TraceSanitizer):
        def validate(self, value, capture_mode):
            raise ValueError("broken")

    failed_sink = _run(sink=BadSink())
    assert failed_sink.result.status == "completed"
    assert failed_sink.persistence.model_dump() == {
        "status": "failed", "error_code": "TRACE_SINK_FAILED",
    }
    recorder = AgentTraceRecorder(
        uuid4(), capture_mode="safe", sanitizer=BadSanitizer(),
    )
    degraded = _run(recorder=recorder)
    assert degraded.result.status == "completed"
    assert degraded.trace.completion_status == "degraded"


def test_original_exception_and_trace_survive_sink_path():
    original = RuntimeError("PRIVATE-ORIGINAL-ERROR")

    class BrokenPlanner:
        def plan(self, request):
            raise original

    sink = InMemoryTraceSink()
    registry = _Registry()
    with pytest.raises(RuntimeError) as raised:
        run_agent_loop_traced(
            UserQuery(query="fail"), BrokenPlanner(), PlanValidator(registry), registry,
            _Writer(), _Verifier(), sink=sink,
        )
    assert raised.value is original
    assert sink.traces[0].completion_status == "failed"
    serialized = sink.traces[0].model_dump_json(exclude_none=True)
    assert "PRIVATE-ORIGINAL-ERROR" not in serialized
    assert "run_failed" in serialized


def test_run_scoped_hmac_is_correlatable_but_not_stable():
    one = TraceProjector("safe", b"a" * 32)
    two = TraceProjector("safe", b"b" * 32)
    assert one.reference("task") == one.reference("task")
    assert one.reference("task") != two.reference("task")


def test_context_indexes_are_trace_only_and_occurrence_safe():
    request = UserQuery(query="q", history=[
        Message(role="user", content="same"), Message(role="user", content="same"),
    ])
    recorder = AgentTraceRecorder(request.request_id, capture_mode="evaluation")
    with recorder.activate(), recorder.scope(operation="context", iteration=0, plan_revision=0):
        selection = ContextManager().select(request, "planner", ContextPolicy(strategy="full_history"))
    assert "selected_message_indexes" not in selection.model_dump()
    event = next(item for item in recorder.events if item.kind == "context_selected")
    assert event.selected_message_indexes == [0, 1]


def test_wide_payload_is_rejected():
    trace = _run(mode="evaluation").trace.model_dump(mode="json")
    trace["events"][0]["payload"] = {"anything": "goes"}
    with pytest.raises(ValidationError):
        AgentTrace.model_validate(trace)


def test_sanitizer_rejects_forged_raw_value_in_safe_trace(tmp_path: Path):
    trace = _run().trace
    forged_event = trace.events[0].model_copy(update={"query": "FORGED-QUERY"})
    forged = trace.model_copy(update={"events": [forged_event, *trace.events[1:]]})
    with pytest.raises(ValueError, match="TRACE_SAFE_RAW_VALUE"):
        JsonlTraceSink(tmp_path / "forged.jsonl").write(forged)


def test_deadline_block_is_traced_without_starting_attempt():
    class DeadlineClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 121.0

    registry = _Registry()
    traced = run_agent_loop_traced(
        UserQuery(query="deadline"), _Planner(), PlanValidator(registry), registry,
        _Writer(), _Verifier(), context=CallContext(), clock=DeadlineClock(),
        capture_mode="evaluation",
    )
    blocked = [item for item in traced.trace.events if item.kind == "tool_attempt_blocked"]
    assert len(blocked) == 1
    assert blocked[0].reason == "deadline"
    assert not [item for item in traced.trace.events if item.kind == "tool_attempt"]
    finished = next(item for item in traced.trace.events if item.kind == "run_finished")
    assert finished.budget.deadline_exceeded


def test_fixed_e2e_set_is_scored_from_unified_trace():
    root = Path(__file__).parents[2]
    scenarios, expectations = load_e2e_eval_set(
        root / "eval" / "agent_e2e" / "scenarios.jsonl",
        root / "eval" / "agent_e2e" / "expectations.jsonl",
    )
    report = evaluate_agent_e2e(scenarios, expectations)
    assert report.all_passed
    wrong_initial = report.cases[6]
    assert wrong_initial.task_success
    assert not wrong_initial.tool_selection_correct
    assert wrong_initial.unnecessary_tool_calls == 1
    exhausted = report.cases[-1]
    events = exhausted.unified_trace.events
    assert len([item for item in events if item.kind == "tool_attempt"]) == 4
    assert [item.plan_revision for item in events if item.kind == "plan_proposed"] == [0, 1, 2]
    blocked = [item for item in events if item.kind == "tool_attempt_blocked"]
    assert len(blocked) == 1 and blocked[0].reason == "attempt_budget"
    assert exhausted.result.stop_reason == "total_tool_budget"
