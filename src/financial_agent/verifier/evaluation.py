"""Fixed-case evaluation and boundary metrics for the structured verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from financial_agent.verifier.models import (
    VerifierEvalCase,
    VerifierEvalMetrics,
    VerifierEvalResult,
    VerifierEvaluationReport,
)
from financial_agent.verifier.service import StructuredVerifier


def load_verifier_eval_cases(path: Path) -> list[VerifierEvalCase]:
    cases = [
        VerifierEvalCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Verifier eval case IDs must be unique")
    return cases


def evaluate_verifier(
    cases: Sequence[VerifierEvalCase],
    verifier: StructuredVerifier,
) -> VerifierEvaluationReport:
    results = []
    for case in cases:
        actual = verifier.verify(case.request, case.plan, case.tool_results, case.draft)
        results.append(VerifierEvalResult(
            case_id=case.case_id,
            expected_decision=case.expected_decision,
            actual=actual,
            correct=actual.decision == case.expected_decision,
        ))
    return VerifierEvaluationReport(metrics=_metrics(results), cases=results)


def _metrics(results: list[VerifierEvalResult]) -> VerifierEvalMetrics:
    predicted_pass = [item for item in results if item.actual.decision == "PASS"]
    rewrite_or_replan = [item for item in results if item.expected_decision in {"REWRITE", "REPLAN"}]
    non_replan = [item for item in results if item.expected_decision != "REPLAN"]
    expected_replan = [item for item in results if item.expected_decision == "REPLAN"]
    cross_confusions = [
        item for item in rewrite_or_replan
        if (item.expected_decision, item.actual.decision) in {
            ("REWRITE", "REPLAN"), ("REPLAN", "REWRITE"),
        }
    ]
    return VerifierEvalMetrics(
        case_count=len(results),
        pass_precision=_ratio(
            sum(item.expected_decision == "PASS" for item in predicted_pass), len(predicted_pass),
        ),
        rewrite_replan_confusion_rate=_ratio(len(cross_confusions), len(rewrite_or_replan)),
        unnecessary_replan_rate=_ratio(
            sum(item.actual.decision == "REPLAN" for item in non_replan), len(non_replan),
        ),
        missed_replan_rate=_ratio(
            sum(item.actual.decision != "REPLAN" for item in expected_replan), len(expected_replan),
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
