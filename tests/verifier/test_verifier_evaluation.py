import json
from pathlib import Path
from uuid import uuid4

import pytest

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.schemas import UserQuery
from financial_agent.tools.contracts import ToolResult
from financial_agent.verifier.evaluation import evaluate_verifier, load_verifier_eval_cases
from financial_agent.verifier.models import (
    DraftAnswer,
    VerificationResult,
    VerifierEvalCase,
)


class DecisionVerifier:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def verify(self, request, plan, tool_results, draft):
        del request, plan, tool_results, draft
        decision = next(self.decisions)
        return VerificationResult(
            decision=decision,
            reason="fixture",
            missing_evidence=["missing"] if decision == "REPLAN" else [],
            failed_task_ids=[],
        )


def case(case_id, expected):
    task = Task(task_id="t1", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="t1", tool_name="lookup",
        result=ToolResult(
            status="success", data={"value": 1}, source="test", latency=0,
            error=None, request_id=uuid4(),
        ),
    )
    return VerifierEvalCase(
        case_id=case_id,
        request=UserQuery(query=case_id),
        plan=[task],
        tool_results=[result],
        draft=DraftAnswer(answer="draft"),
        expected_decision=expected,
    )


def test_verifier_eval_reports_boundary_metrics():
    cases = [
        case("p1", "PASS"), case("p2", "PASS"),
        case("w1", "REWRITE"), case("w2", "REWRITE"),
        case("r1", "REPLAN"), case("r2", "REPLAN"),
    ]
    actual = ["PASS", "REWRITE", "REPLAN", "REWRITE", "REWRITE", "PASS"]

    report = evaluate_verifier(cases, DecisionVerifier(actual))

    assert report.metrics.case_count == 6
    assert report.metrics.pass_precision == 0.5
    assert report.metrics.rewrite_replan_confusion_rate == 0.5
    assert report.metrics.unnecessary_replan_rate == 0.25
    assert report.metrics.missed_replan_rate == 1.0


def test_eval_loader_rejects_duplicate_case_ids(tmp_path: Path):
    payload = case("same", "PASS").model_dump(mode="json")
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join([json.dumps(payload), json.dumps(payload)]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        load_verifier_eval_cases(path)
