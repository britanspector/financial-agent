from hashlib import sha256
from pathlib import Path

import pytest

from financial_agent.agent.e2e_evaluation import evaluate_agent_e2e, load_e2e_eval_set


ROOT = Path(__file__).parents[2]
SCENARIOS = ROOT / "eval" / "agent_e2e" / "scenarios.jsonl"
EXPECTATIONS = ROOT / "eval" / "agent_e2e" / "expectations.jsonl"
SCENARIOS_SHA256 = "21ca0b25caaecea7f5e538855fa8fa27e9966991414748d60e745647213d6ea8"
EXPECTATIONS_SHA256 = "64a902035d11c32bed9663037436a161c153c992373591dbff64e7b6a5ea97a2"


def _load():
    return load_e2e_eval_set(SCENARIOS, EXPECTATIONS)


def test_fixed_phase61_dataset_and_metrics_are_deterministic():
    assert sha256(SCENARIOS.read_bytes()).hexdigest() == SCENARIOS_SHA256
    assert sha256(EXPECTATIONS.read_bytes()).hexdigest() == EXPECTATIONS_SHA256
    scenarios, expectations = _load()

    report = evaluate_agent_e2e(scenarios, expectations)

    assert report.case_count == 12
    assert report.passed_count == 12
    assert report.all_passed
    assert report.metrics.task_success_rate == 1
    assert report.metrics.final_answer_correctness == 1
    assert report.metrics.tool_selection_accuracy == pytest.approx(11 / 12)
    assert report.metrics.unnecessary_tool_call_rate == pytest.approx(1 / 18)
    assert report.metrics.replan_recovery_rate == 1
    assert report.metrics.average_tool_attempts == pytest.approx(22 / 12)
    assert report.metrics.loop_iterations == pytest.approx(16 / 12)


def test_wrong_initial_plan_recovers_without_hiding_unnecessary_tool():
    report = evaluate_agent_e2e(*_load())
    case = next(item for item in report.cases if item.case_id == "wrong_plan_replan_recovery")

    assert case.passed
    assert case.task_success
    assert case.result.replan_count == 1
    assert [item.action for item in case.plan_observations] == ["initial", "replan"]
    assert not case.tool_selection_correct
    assert case.unnecessary_tool_calls == 1
    assert {item.tool_name for item in case.actual_logical_tools} == {
        "get_market_snapshot", "get_portfolio_analytics",
    }


def test_retry_and_replan_share_explicit_four_attempt_budget():
    report = evaluate_agent_e2e(*_load())
    case = next(item for item in report.cases if item.case_id == "shared_budget_exhausted")

    assert case.passed
    assert case.result.status == "limit_exhausted"
    assert case.result.stop_reason == "total_tool_budget"
    assert case.result.total_tool_attempts == 4
    assert case.result.replan_count == 2
    assert [item.error_code for item in case.tool_invocations] == [
        "TEMPORARY_FAILURE", None, "TEMPORARY_FAILURE", "TEMPORARY_FAILURE",
    ]
    assert {item.tool_name for item in case.actual_logical_tools} == {
        "get_customer_context", "get_portfolio_analytics",
    }


def test_context_cases_record_only_required_retrieval_diagnostics():
    report = evaluate_agent_e2e(*_load())
    cases = [item for item in report.cases if item.case_id.startswith("history_retrieval_")]

    assert len(cases) == 2
    assert all(any(item.retrieval_active for item in case.context_observations) for case in cases)
    assert all(any(0 in item.retrieved_message_indexes for item in case.context_observations) for case in cases)
    assert set(type(cases[0].context_observations[0]).model_fields) == {
        "component", "retrieval_active", "retrieved_message_indexes",
    }


def test_scorer_detects_wrong_answer_tool_gold_and_retrieval_requirement():
    scenarios, expectations = _load()
    changed = list(expectations)
    changed[0] = changed[0].model_copy(update={"required_text": ["not present"]})
    answer_report = evaluate_agent_e2e(scenarios, changed)
    assert answer_report.metrics.final_answer_correctness < 1
    assert not answer_report.cases[0].passed

    changed = list(expectations)
    changed[1] = changed[1].model_copy(update={"expected_tools": changed[1].expected_tools[:1]})
    tool_report = evaluate_agent_e2e(scenarios, changed)
    assert tool_report.metrics.tool_selection_accuracy < 11 / 12
    assert tool_report.cases[1].unnecessary_tool_calls == 1

    changed = list(expectations)
    changed[8] = changed[8].model_copy(update={"required_retrieved_message_indexes": [99]})
    context_report = evaluate_agent_e2e(scenarios, changed)
    assert not context_report.cases[8].passed
    assert "context_retrieval" in context_report.cases[8].failures


def test_loader_rejects_misaligned_case_ids(tmp_path):
    scenarios = tmp_path / "scenarios.jsonl"
    expectations = tmp_path / "expectations.jsonl"
    scenarios.write_text(SCENARIOS.read_text(encoding="utf-8"), encoding="utf-8")
    expectations.write_text(
        EXPECTATIONS.read_text(encoding="utf-8").replace(
            '"case_id":"single_tool_pass"', '"case_id":"different"', 1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identical ordered case IDs"):
        load_e2e_eval_set(scenarios, expectations)
