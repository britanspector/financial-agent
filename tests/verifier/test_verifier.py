from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.verifier.models import DraftAnswer, EvidenceReference, VerifierModelOutput
from financial_agent.verifier.providers import VerifierProviderResponseError
from financial_agent.verifier.service import InvalidVerificationInputError, StructuredVerifier


class Item(Schema):
    value: int


class Output(Schema):
    summary: str
    items: list[Item]


class Catalog:
    def output_model(self, name):
        return Output if name == "lookup" else None


class FakeProvider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return self.output


def task(task_id="t1"):
    return Task(task_id=task_id, tool_name="lookup", arguments={}, dependencies=[])


def result(task_id="t1", *, status="success", code="FAIL"):
    error = ToolError(code=code, message=code, http_status=503, retryable=False) if status == "error" else None
    return TaskExecutionResult(
        task_id=task_id,
        tool_name="lookup",
        result=ToolResult(
            status=status,
            data=None if error else {"summary": "found", "items": [{"value": 7}]},
            source="test",
            latency=12,
            error=error,
            request_id=uuid4(),
        ),
        max_retry=2,
    )


def output(decision="PASS", missing=None):
    return {
        "decision": decision,
        "reason": "Evidence sufficiency checked",
        "missing_evidence": missing or [],
    }


def test_verifier_sends_complete_inputs_and_resolved_safe_evidence():
    provider = FakeProvider(output())
    verifier = StructuredVerifier(provider, Catalog())
    request = UserQuery(
        query="What is the value?",
        history=[Message(role="user", content="Use the latest lookup")],
    )
    draft = DraftAnswer(
        answer="The value is 7.",
        evidence=[EvidenceReference(task_id="t1", source_path=["items", 0, "value"])],
    )

    verified = verifier.verify(request, [task()], [result()], draft)

    payload = json.loads(provider.calls[0][0][1]["content"])
    assert verified.decision == "PASS"
    assert verified.failed_task_ids == []
    assert payload["query"] == request.query
    assert payload["history"][0]["content"] == "Use the latest lookup"
    assert payload["draft"]["evidence"][0]["value"] == 7
    assert set(payload["tool_results"][0]) == {
        "task_id", "tool_name", "status", "data", "source", "error", "retry_count", "max_retry",
    }
    assert "request_id" not in provider.calls[0][0][1]["content"]
    assert "latency" not in provider.calls[0][0][1]["content"]
    assert all(
        "failed_task_ids" not in branch["properties"]
        for branch in provider.calls[0][1]["oneOf"]
    )


def test_response_schema_constrains_missing_evidence_per_decision():
    provider = FakeProvider(output())
    StructuredVerifier(provider, Catalog()).verify(
        UserQuery(query="verify"), [task()], [result()], DraftAnswer(answer="draft"),
    )
    branches = provider.calls[0][1]["oneOf"]
    by_decision = {branch["properties"]["decision"]["const"]: branch for branch in branches}
    assert by_decision["PASS"]["properties"]["missing_evidence"]["maxItems"] == 0
    assert by_decision["REWRITE"]["properties"]["missing_evidence"]["maxItems"] == 0
    assert by_decision["REPLAN"]["properties"]["missing_evidence"]["minItems"] == 1


def test_failed_task_ids_are_computed_in_plan_order_not_generated_by_model():
    provider = FakeProvider(output("REPLAN", ["Required lookups failed"]))
    verifier = StructuredVerifier(provider, Catalog())
    plan = [task("first"), task("second")]
    results = [result("second", status="error"), result("first", status="error")]

    verified = verifier.verify(
        UserQuery(query="Need both facts"), plan, results,
        DraftAnswer(answer="Unable to answer."),
    )

    assert verified.failed_task_ids == ["first", "second"]
    payload = json.loads(provider.calls[0][0][1]["content"])
    assert payload["failed_task_ids"] == ["first", "second"]


@pytest.mark.parametrize(
    ("plan", "results", "message"),
    [
        ([], [], "non-empty execute plan"),
        ([task()], [], "must match exactly"),
        ([task()], [result(), result()], "must be unique"),
        ([task()], [result("extra")], "must match exactly"),
        ([task()], [TaskExecutionResult(
            task_id="t1", tool_name="other", result=result().result,
        )], "names must match"),
    ],
)
def test_invalid_plan_result_alignment_fails_before_provider(plan, results, message):
    provider = FakeProvider(output())
    with pytest.raises(InvalidVerificationInputError, match=message):
        StructuredVerifier(provider, Catalog()).verify(
            UserQuery(query="verify"), plan, results, DraftAnswer(answer="draft"),
        )
    assert provider.calls == []


@pytest.mark.parametrize(
    ("reference", "tool_result", "message"),
    [
        (EvidenceReference(task_id="missing", source_path=["summary"]), result(), "does not exist"),
        (EvidenceReference(task_id="t1", source_path=["summary"]), result(status="error"), "task failed"),
        (EvidenceReference(task_id="t1", source_path=["private"]), result(), "not a public result field"),
        (EvidenceReference(task_id="t1", source_path=["items", 5, "value"]), result(), "value is unavailable"),
    ],
)
def test_invalid_evidence_reference_fails_before_provider(reference, tool_result, message):
    provider = FakeProvider(output())
    with pytest.raises(InvalidVerificationInputError, match=message):
        StructuredVerifier(provider, Catalog()).verify(
            UserQuery(query="verify"), [task()], [tool_result],
            DraftAnswer(answer="draft", evidence=[reference]),
        )
    assert provider.calls == []


def test_negative_evidence_index_is_rejected_by_contract():
    with pytest.raises(ValidationError, match="non-negative"):
        EvidenceReference(task_id="t1", source_path=["items", -1])


def test_boolean_evidence_index_is_rejected_by_contract():
    with pytest.raises(ValidationError):
        EvidenceReference(task_id="t1", source_path=["items", True])


@pytest.mark.parametrize(
    ("decision", "missing"),
    [("PASS", ["new fact"]), ("REWRITE", ["new fact"]), ("REPLAN", [])],
)
def test_decision_and_missing_evidence_are_consistent(decision, missing):
    with pytest.raises(ValidationError):
        VerifierModelOutput(decision=decision, reason="reason", missing_evidence=missing)


def test_model_cannot_return_failed_task_ids_or_extra_fields():
    provider = FakeProvider({**output(), "failed_task_ids": ["invented"]})
    with pytest.raises(VerifierProviderResponseError, match="Invalid verifier response"):
        StructuredVerifier(provider, Catalog()).verify(
            UserQuery(query="verify"), [task()], [result()], DraftAnswer(answer="draft"),
        )


def test_prompt_defines_rewrite_replan_boundary_by_evidence_not_writing_quality():
    provider = FakeProvider(output("REWRITE"))
    StructuredVerifier(provider, Catalog()).verify(
        UserQuery(query="verify"), [task()], [result()], DraftAnswer(answer="poor draft"),
    )
    rules = provider.calls[0][0][0]["content"]
    assert "already contain all necessary evidence" in rules
    assert "Poor writing alone is never a reason to REPLAN" in rules
    assert "results themselves lack necessary evidence" in rules
    assert "If even one requested fact is absent" in rules
    assert "decision must be REPLAN, never REWRITE" in rules
