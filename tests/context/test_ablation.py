from hashlib import sha256
from pathlib import Path

from financial_agent.context.ablation import evaluate_context_ablation, load_ablation_cases


ROOT = Path(__file__).parents[2]
HOLDOUT = ROOT / "eval" / "context_ablation" / "holdout_cases.jsonl"
EXPECTED_SHA256 = "8db13562ebbe11dffcb9a218f10e90011d902fc92c5182d8eb16bb86827a3344"


def test_phase54_holdout_is_fixed_and_offline_hard_gates_pass():
    payload = HOLDOUT.read_bytes()
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
    assert retrieval.task_accuracy >= report.strategies[1].task_accuracy
    assert retrieval.summary_incremental_update_count >= 1
    assert retrieval.summary_input_tokens < retrieval.repeated_rebuild_counterfactual_tokens
    # Experimental targets are always reported, but are not promoted to correctness gates.
    assert isinstance(report.experimental_targets["history_token_ratio_le_0_60"], bool)
