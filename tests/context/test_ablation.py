from hashlib import sha256
from pathlib import Path

from financial_agent.context.ablation import (
    evaluate_context_ablation,
    load_ablation_cases,
    rescore_ablation_report,
    semantic_arguments_match,
)


ROOT = Path(__file__).parents[2]
HOLDOUT = ROOT / "eval" / "context_ablation" / "holdout_cases.jsonl"
EXPECTED_SHA256 = "8db13562ebbe11dffcb9a218f10e90011d902fc92c5182d8eb16bb86827a3344"


def test_phase54_holdout_is_fixed_and_offline_hard_gates_pass():
    payload = HOLDOUT.read_text(encoding="utf-8").encode("utf-8")
    assert sha256(payload).hexdigest() == EXPECTED_SHA256

    report = evaluate_context_ablation(
        load_ablation_cases(HOLDOUT),
        holdout_sha256=EXPECTED_SHA256,
    )

    assert report.case_count == 12
    assert report.hard_gate_passed
    assert report.hard_gate_failures == []
    assert [item.strategy for item in report.strategies] == [
        "full_history",
        "last_n",
        "budgeted_selection",
        "summary_compression",
        "summary_retrieval",
    ]
    retrieval = report.strategies[-1]
    assert retrieval.protected_retention == 1
    assert retrieval.planner_strict_accuracy == 1
    assert retrieval.planner_semantic_accuracy == 1
    assert retrieval.task_accuracy >= report.strategies[1].task_accuracy
    assert retrieval.summary_incremental_update_count >= 1
    assert retrieval.summary_input_tokens < retrieval.repeated_rebuild_counterfactual_tokens
    # Experimental targets are always reported, but are not promoted to correctness gates.
    assert isinstance(report.experimental_targets["history_token_ratio_le_0_60"], bool)
    relative = {item.strategy: item for item in report.relative_strategies}
    assert relative["summary_retrieval"].planner.full_history_preservation_rate == 1
    assert relative["summary_retrieval"].planner.context_induced_regression_count == 0
    assert relative["summary_retrieval"].planner.recovery_count == 0
    assert relative["last_n"].planner.context_induced_regression_count == 12
    assert len(report.case_diagnostics) == 12 * 5


def test_semantic_scorer_maps_open_topic_but_keeps_structured_fields_strict():
    expected = {
        "topic": "cashflow_risk",
        "entity": "霁川能源",
        "start_date": None,
        "end_date": "2026-07-15",
        "constraint": None,
    }

    assert semantic_arguments_match(expected, {
        **expected, "topic": "现金流断裂风险",
    })
    assert not semantic_arguments_match(expected, {
        **expected, "topic": "现金流断裂风险", "entity": "另一家公司",
    })
    assert not semantic_arguments_match(expected, {
        **expected, "topic": "现金流断裂风险", "end_date": "2026-07-16",
    })
    assert not semantic_arguments_match(expected, {
        **expected, "topic": "现金流断裂风险", "extra": "not allowed",
    })


def test_semantic_scorer_normalizes_explicit_constraint_aliases():
    expected = {
        "topic": "market_snapshot", "entity": "远岑科技",
        "start_date": None, "end_date": None, "constraint": "现在可以查询行情",
    }
    actual = {**expected, "topic": "行情快照", "constraint": "允许查询行情"}

    assert semantic_arguments_match(expected, actual)


def test_report_can_be_rescored_from_persisted_actual_arguments_without_model_calls():
    cases = load_ablation_cases(HOLDOUT)
    report = evaluate_context_ablation(cases, holdout_sha256=EXPECTED_SHA256)
    target_index = next(
        index for index, run in enumerate(report.runs)
        if run.case_id == "holdout_assistant_cashflow" and run.strategy == "full_history"
    )
    run = report.runs[target_index]
    actual = {**run.actual_arguments, "topic": "现金流断裂风险"}
    changed = run.model_copy(update={
        "actual_arguments": actual,
        "planner_correct": False,
        "planner_strict_correct": False,
        "planner_semantic_correct": False,
    })
    runs = list(report.runs)
    runs[target_index] = changed

    rescored = rescore_ablation_report(report.model_copy(update={"runs": runs}), cases)
    rescored_run = rescored.runs[target_index]

    assert not rescored_run.planner_strict_correct
    assert rescored_run.planner_semantic_correct


def test_relative_analysis_separates_planner_success_from_downstream_infra_failure():
    cases = load_ablation_cases(HOLDOUT)
    report = evaluate_context_ablation(cases, holdout_sha256=EXPECTED_SHA256)
    case = cases[0]
    target_index = next(
        index for index, run in enumerate(report.runs)
        if run.case_id == case.case_id and run.strategy == "last_n"
    )
    run = report.runs[target_index]
    runs = list(report.runs)
    runs[target_index] = run.model_copy(update={
        "actual_tool_name": "lookup_financial_fact",
        "actual_arguments": case.expected_arguments,
        "failure": "AnswerProviderResponseError",
    })

    rescored = rescore_ablation_report(report.model_copy(update={"runs": runs}), cases)
    diagnostic = next(
        item for item in rescored.case_diagnostics
        if item.case_id == case.case_id and item.strategy == "last_n"
    )

    assert diagnostic.planner_comparison == "preserved"
    assert diagnostic.task_comparison == "infra_failure"
    assert diagnostic.failure_reason == "AnswerProviderResponseError"
