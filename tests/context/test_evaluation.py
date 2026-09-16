from pathlib import Path

import pytest

from financial_agent.context.evaluation import (
    ContextEvalCase,
    evaluate_context_selection,
    load_context_eval_cases,
    report_summary,
)


CASES = Path(__file__).parents[2] / "eval" / "context" / "context_cases.jsonl"


def test_fixed_context_eval_cases_are_unique_and_cover_expected_scenarios():
    cases = load_context_eval_cases(CASES)
    assert len(cases) == 15
    assert len({case.case_id for case in cases}) == 15
    assert all(len(case.history) >= 8 for case in cases)
    assert {item.kind for case in cases for item in case.critical_context} == {
        "entity", "constraint", "context",
    }
    assert all(
        item.hard_required
        for case in cases for item in case.critical_context
        if item.kind in {"entity", "constraint"}
    )


def test_context_eval_reports_five_phase5_baseline_metrics_per_strategy():
    report = evaluate_context_selection(load_context_eval_cases(CASES))
    assert [item.strategy for item in report.strategies] == [
        "full_history", "last_n", "budgeted_selection", "summary_compression", "summary_retrieval",
    ]
    assert all(item.case_count == 15 for item in report.strategies)
    assert report.summary_baseline_passed
    assert report.retrieval_baseline_passed
    assert report.recovered_case_count_vs_summary == 3
    assert report.summary_failed_case_ids == []
    assert report.strategies[0].critical_context_retention_rate == 1
    assert report.strategies[0].history_token_compression_ratio == 1
    assert report.strategies[0].planner_plan_equivalence_rate == 1
    assert report.strategies[0].critical_entity_or_constraint_loss_rate == 0
    assert set(report_summary(report)["strategies"][0]) >= {
        "critical_context_retention_rate",
        "history_token_compression_ratio",
        "planner_plan_equivalence_rate",
        "critical_entity_or_constraint_loss_rate",
        "hard_fact_retention_rate",
        "retrieval_hit_rate",
        "retrieval_recall",
    }
    assert report_summary(report)["summary_baseline_passed"] is True


def test_budgeted_baseline_has_no_catastrophic_information_loss():
    report = evaluate_context_selection(load_context_eval_cases(CASES))
    metrics = {item.strategy: item for item in report.strategies}
    budgeted = metrics["budgeted_selection"]
    last_n = metrics["last_n"]

    assert budgeted.critical_context_retention_rate >= 0.70
    assert budgeted.planner_plan_equivalence_rate >= 0.70
    assert budgeted.critical_entity_or_constraint_loss_rate <= 0.30
    assert budgeted.history_token_compression_ratio < 1
    assert budgeted.critical_context_retention_rate >= last_n.critical_context_retention_rate


def test_fixed_strategy_baseline_values_are_stable():
    report = evaluate_context_selection(load_context_eval_cases(CASES))
    metrics = {item.strategy: item for item in report.strategies}

    assert metrics["last_n"].critical_context_retention_rate == pytest.approx(0.3846153846)
    assert metrics["last_n"].history_token_compression_ratio == pytest.approx(0.5662544170)
    assert metrics["last_n"].planner_plan_equivalence_rate == pytest.approx(4 / 15)
    assert metrics["last_n"].critical_entity_or_constraint_loss_rate == pytest.approx(5 / 9)
    assert metrics["budgeted_selection"].critical_context_retention_rate == 1
    assert metrics["budgeted_selection"].history_token_compression_ratio == pytest.approx(0.8648409894)
    assert metrics["budgeted_selection"].planner_plan_equivalence_rate == 1
    assert metrics["budgeted_selection"].critical_entity_or_constraint_loss_rate == 0
    assert metrics["summary_compression"].critical_context_retention_rate == pytest.approx(23 / 26)
    assert metrics["summary_compression"].history_token_compression_ratio == pytest.approx(0.8599823322)
    assert metrics["summary_compression"].planner_plan_equivalence_rate == pytest.approx(0.8)
    assert metrics["summary_compression"].critical_entity_or_constraint_loss_rate == 0
    assert all(
        item.passed for item in report.cases if item.strategy == "summary_compression"
    )
    retrieval = metrics["summary_retrieval"]
    assert retrieval.critical_context_retention_rate == 1
    assert retrieval.planner_plan_equivalence_rate == 1
    assert retrieval.hard_fact_retention_rate == 1
    assert retrieval.retrieval_recall == 1
    assert retrieval.critical_entity_or_constraint_loss_rate == 0
    assert retrieval.history_token_compression_ratio < 1
    assert all(
        not item.metrics.retrieval_active or item.metrics.retrieved_turn_count > 0
        for item in report.cases
        if item.strategy == "summary_retrieval"
    )


def test_empty_eval_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_context_selection([])


def test_case_file_size_guard_is_enforced(tmp_path):
    case = ContextEvalCase.model_validate({
        "case_id": "one",
        "query": "q",
        "history": [{"role": "user", "content": "history"}],
        "critical_context": [{"key": "key", "value": "history", "kind": "context"}],
    })
    path = tmp_path / "too-small.jsonl"
    path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="10 to 15"):
        load_context_eval_cases(path)
