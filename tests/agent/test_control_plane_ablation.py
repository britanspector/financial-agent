from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from financial_agent.agent.control_plane_ablation import (
    CombinedManifest, NoHistoryContextAdapter, _attribution_evidence, _failure_type,
    evaluate_control_plane_ablation, load_control_plane_ablation,
)
from financial_agent.observability import AgentTrace, JsonlTraceSink, load_agent_traces
from financial_agent.observability.recorder import AgentTraceRecorder
from financial_agent.schemas import Message, UserQuery


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "eval" / "control_plane_ablation" / "combined_manifest.json"
MANIFEST_SHA256 = "5d9777afbcf5added7c08b54bdffdfb7f4a0224d80a5d1298c343836f8f14336"
PHASE61_SCENARIOS_SHA256 = "21ca0b25caaecea7f5e538855fa8fa27e9966991414748d60e745647213d6ea8"
PHASE61_EXPECTATIONS_SHA256 = "64a902035d11c32bed9663037436a161c153c992373591dbff64e7b6a5ea97a2"


@pytest.fixture(scope="module")
def eval_set():
    return load_control_plane_ablation(MANIFEST)


@pytest.fixture(scope="module")
def report(eval_set):
    return evaluate_control_plane_ablation(eval_set)


def test_combined_manifest_and_phase61_files_are_locked(eval_set):
    assert sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256
    assert sha256((ROOT / "eval/agent_e2e/scenarios.jsonl").read_bytes()).hexdigest() == PHASE61_SCENARIOS_SHA256
    assert sha256((ROOT / "eval/agent_e2e/expectations.jsonl").read_bytes()).hexdigest() == PHASE61_EXPECTATIONS_SHA256
    assert eval_set.manifest_hash == MANIFEST_SHA256
    assert len(eval_set.scenarios) == len(eval_set.expectations) == 16
    assert [item.case_id for item in eval_set.scenarios[-4:]] == [
        "retry_only_recovery", "rewrite_only_recovery",
        "replan_only_recovery", "context_only_recovery",
    ]


def test_five_configuration_matrix_and_recovery_staircase(report):
    assert report.all_passed
    assert report.case_count == 16
    assert report.configuration_count == 5
    assert report.run_count == 80
    assert len(report.deltas) == 4
    assert report.phase61_full_compatible
    assert report.recovery_staircase_valid
    assert [item.recovery_rate for item in report.metrics] == [0, 0.25, 0.5, 0.75, 1]
    assert [item.recovery_rate_delta for item in report.deltas] == [0.25] * 4
    assert all(item.latency_is_small_sample_descriptive for item in report.metrics)


def test_full_agent_preserves_phase61_submatrix(report):
    original = [item for item in report.cells if item.configuration == "full_agent"][:12]
    assert len(original) == 12
    assert sum(item.task_success for item in original if item.solvable) == 11
    assert sum(item.tool_attempts for item in original) == 22
    assert sum(item.loop_iterations for item in original) == 16
    assert sum(item.tool_selection_correct for item in original) == 11
    wrong = next(item for item in original if item.case_id == "wrong_plan_replan_recovery")
    assert wrong.task_success
    assert not wrong.tool_selection_correct
    assert wrong.unnecessary_tool_calls == 1


def test_trace_proves_incremental_capability_boundaries(report):
    simple = [item for item in report.cells if item.configuration == "simple_baseline"]
    assert not any(event.kind == "retry_scheduled" for cell in simple for event in cell.trace.events)
    assert not any(event.kind == "verifier_completed" for cell in simple for event in cell.trace.events)
    assert not any((event.plan_revision or 0) > 0 for cell in simple for event in cell.trace.events)
    assert not any(getattr(event, "retrieval_active", False) for cell in simple for event in cell.trace.events)
    assert all(
        event.metrics.selected_message_count == 0
        for cell in simple for event in cell.trace.events if event.kind == "context_selected"
    )
    assert any(event.kind == "retry_scheduled" for cell in report.cells
               if cell.configuration == "retry" for event in cell.trace.events)
    assert any(event.kind == "writer_completed" and event.mode == "rewrite"
               for cell in report.cells if cell.configuration == "verifier_rewrite"
               for event in cell.trace.events)
    assert any((event.plan_revision or 0) > 0 for cell in report.cells
               if cell.configuration == "replan" for event in cell.trace.events)
    assert any(getattr(event, "retrieval_active", False) for cell in report.cells
               if cell.configuration == "full_agent" for event in cell.trace.events)


def test_mechanism_attribution_uses_typed_event_sequences(report, eval_set):
    assert [item.status for item in report.attributions] == ["credited"] * 4
    assert [len(item.evidence) for item in report.attributions] == [3, 3, 3, 2]
    assert [[pointer.kind for pointer in item.evidence] for item in report.attributions] == [
        ["tool_attempt", "retry_scheduled", "tool_attempt"],
        ["verifier_completed", "writer_completed", "verifier_completed"],
        ["verifier_completed", "plan_proposed", "verifier_completed"],
        ["context_selected", "plan_proposed"],
    ]
    retry_after = next(item for item in report.cells
                       if item.configuration == "retry" and item.case_id == "retry_only_recovery")
    retry_before = next(item for item in report.cells
                        if item.configuration == "simple_baseline" and item.case_id == "retry_only_recovery")
    expectation = next(item for item in eval_set.expectations if item.case_id == "retry_only_recovery")
    stripped = retry_after.model_copy(update={
        "trace": retry_after.trace.model_copy(update={
            "events": [event for event in retry_after.trace.events if event.kind != "retry_scheduled"],
        }),
    })
    assert _attribution_evidence("retry", retry_before, stripped, expectation) == []


def test_trace_health_does_not_override_agent_failure(report):
    budget = next(item for item in report.cells
                  if item.configuration == "full_agent" and item.case_id == "shared_budget_exhausted")
    assert budget.run_failure_type == "tool_budget_exhausted"
    assert budget.trace_health == "healthy"
    degraded = budget.trace.model_copy(update={
        "dropped_event_count": 1, "recorder_error_codes": ["TRACE_EVENT_DROPPED"],
        "completion_status": "degraded",
    })
    assert _failure_type(budget.result, degraded, budget.answer_correct, None) == "tool_budget_exhausted"


def test_context_metric_is_total_with_component_breakdown(report):
    for cell in report.cells:
        assert cell.total_selected_history_tokens == sum(
            cell.selected_history_tokens_by_component.values()
        )
    full_context = next(item for item in report.cells
                        if item.configuration == "full_agent" and item.case_id == "context_only_recovery")
    off_context = next(item for item in report.cells
                       if item.configuration == "replan" and item.case_id == "context_only_recovery")
    assert off_context.total_selected_history_tokens == 0
    assert full_context.selected_history_tokens_by_component["planner"] > 0


def test_no_history_adapter_explicitly_removes_history():
    request = UserQuery(query="current", history=[Message(role="user", content="old")])
    recorder = AgentTraceRecorder(request.request_id, capture_mode="evaluation")
    with recorder.activate():
        selection = NoHistoryContextAdapter().select(request, "planner", object())
    assert selection.request.query == "current"
    assert selection.request.history == []
    event = next(item for item in recorder.events if item.kind == "context_selected")
    assert event.selected_message_indexes == []
    assert event.metrics.selected_message_count == 0


def test_jsonl_contains_one_trace_per_matrix_cell(tmp_path, eval_set):
    path = tmp_path / "ablation.jsonl"
    report = evaluate_control_plane_ablation(eval_set, trace_sink=JsonlTraceSink(path))
    traces = load_agent_traces(path)
    assert report.all_passed
    assert len(traces) == report.run_count == 80
    assert len({item.trace_id for item in traces}) == 80


def test_loader_rejects_hash_mismatch_and_manifest_schema(tmp_path):
    root = tmp_path
    for relative in (
        "eval/agent_e2e/scenarios.jsonl", "eval/agent_e2e/expectations.jsonl",
        "eval/control_plane_ablation/scenarios.jsonl",
        "eval/control_plane_ablation/expectations.jsonl",
        "eval/control_plane_ablation/counterfactuals.jsonl",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["sources"][0]["sha256"] = "0" * 64
    manifest = root / "eval/control_plane_ablation/combined_manifest.json"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_control_plane_ablation(manifest)
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["cases"].append(raw["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        CombinedManifest.model_validate(raw)
