from pathlib import Path

from financial_agent.verifier.evaluation import load_verifier_eval_cases


def test_fixed_eval_set_covers_all_verdicts_and_unique_cases():
    path = Path(__file__).parents[2] / "eval" / "verifier" / "verifier_cases.jsonl"
    cases = load_verifier_eval_cases(path)

    assert len(cases) == 9
    assert {case.expected_decision for case in cases} == {"PASS", "REWRITE", "REPLAN"}
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.tool_results[0].result.status == "error" for case in cases)
