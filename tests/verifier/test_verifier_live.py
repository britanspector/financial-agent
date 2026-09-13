import os
from pathlib import Path

import pytest

from financial_agent.config import Settings
from financial_agent.schemas import Schema
from financial_agent.verifier.evaluation import evaluate_verifier, load_verifier_eval_cases
from financial_agent.verifier.runtime import build_verifier


class EvalOutput(Schema):
    value: str


class EvalCatalog:
    def output_model(self, name):
        return EvalOutput if name == "lookup_fact" else None


@pytest.mark.live
def test_live_qwen_verifier_fixed_eval():
    if not os.getenv("FINANCIAL_AGENT_QWEN_API_KEY"):
        pytest.skip("FINANCIAL_AGENT_QWEN_API_KEY is not configured")
    path = Path(__file__).parents[2] / "eval" / "verifier" / "verifier_cases.jsonl"
    report = evaluate_verifier(load_verifier_eval_cases(path), build_verifier(Settings(), EvalCatalog()))

    assert report.metrics.case_count == 9
    assert report.metrics.pass_precision >= 0.5
    assert report.metrics.unnecessary_replan_rate <= 0.5
    assert report.metrics.missed_replan_rate <= 0.5
