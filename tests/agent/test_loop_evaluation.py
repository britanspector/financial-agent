from pathlib import Path

from financial_agent.agent.loop_evaluation import evaluate_loop_cases, load_loop_eval_cases


CASES = Path(__file__).parents[2] / "eval" / "loop" / "loop_cases.jsonl"


def test_fixed_loop_eval_covers_required_black_box_scenarios():
    cases = load_loop_eval_cases(CASES)
    report = evaluate_loop_cases(cases)

    assert [case.case_id for case in cases] == [
        "initial_pass", "rewrite_then_pass", "replan_add_tool", "replan_modify_task",
        "empty_force_rerun", "error_replan_recovery", "partial_result_reuse",
        "binding_upstream_change", "budget_exhausted", "no_progress", "max_iterations", "max_rewrite",
        "max_replan", "fifth_verify_pass", "retry_and_replan_share_budget",
    ]
    assert report.case_count == 15
    assert report.passed_count == 15, {
        item.case_id: item.failures for item in report.cases if not item.passed
    }
    assert report.all_passed


def test_rewrite_eval_proves_same_results_and_zero_tool_calls():
    report = evaluate_loop_cases(load_loop_eval_cases(CASES))
    rewrite_cases = [item for item in report.cases if item.actual.rewrite_count]
    assert rewrite_cases
    assert all(item.rewrite_is_tool_free for item in rewrite_cases)
    assert all(item.rewrite_preserves_results for item in rewrite_cases)
