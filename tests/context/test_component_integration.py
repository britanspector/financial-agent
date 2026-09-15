from __future__ import annotations

import json
from uuid import uuid4

import pytest

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.answering.runtime import build_answer_writer
from financial_agent.config import Settings
from financial_agent.context import ContextManager, HeuristicTokenEstimator, build_context_manager
from financial_agent.planner.runtime import build_planner
from financial_agent.schemas import Message, Schema, UserQuery
from financial_agent.tools.contracts import ToolResult
from financial_agent.verifier.models import DraftAnswer, VerificationResult
from financial_agent.verifier.runtime import build_verifier


class Input(Schema):
    value: int


class Output(Schema):
    value: int


class Catalog:
    def describe(self):
        return [{"name": "lookup", "description": "lookup", "input_schema": Input.model_json_schema()}]

    def input_model(self, name):
        return Input if name == "lookup" else None

    def output_model(self, name):
        return Output if name == "lookup" else None


class Provider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return self.output


class UnitEstimator:
    def estimate_messages(self, messages):
        return len(messages)


def request():
    return UserQuery(
        query="current",
        history=[
            Message(role="user", content="old"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="latest"),
        ],
    )


def inputs():
    task = Task(task_id="t1", tool_name="lookup", arguments={"value": 7})
    result = TaskExecutionResult(
        task_id="t1",
        tool_name="lookup",
        result=ToolResult(
            status="success",
            data={"value": 7},
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    return task, result


def payload(provider):
    return json.loads(provider.calls[-1][0][1]["content"])


def test_all_components_apply_context_before_existing_prompt_builders():
    settings = Settings(context_strategy="last_n", context_last_n=1, qwen_api_key=None)
    catalog = Catalog()
    manager = ContextManager()

    planner_provider = Provider({"decision": "no_tool", "tasks": []})
    planner, _ = build_planner(
        settings, catalog, provider=planner_provider, context_manager=manager,
    )
    planner.plan(request())

    task, result = inputs()
    writer_provider = Provider({
        "answer": "7",
        "evidence": [{"task_id": "t1", "source_path": ["value"]}],
    })
    writer = build_answer_writer(
        settings, catalog, provider=writer_provider, context_manager=manager,
    )
    draft = writer.write(request(), [task], [result])

    verifier_provider = Provider({"decision": "PASS", "reason": "enough", "missing_evidence": []})
    verifier = build_verifier(
        settings, catalog, provider=verifier_provider, context_manager=manager,
    )
    verifier.verify(request(), [task], [result], draft)

    assert payload(planner_provider)["history"] == [{"role": "user", "content": "latest"}]
    assert payload(writer_provider)["history"] == [{"role": "user", "content": "latest"}]
    assert payload(verifier_provider)["history"] == [{"role": "user", "content": "latest"}]
    assert planner._context_manager is writer._context_manager is verifier._context_manager


def test_runtime_maps_separate_component_budgets_and_keeps_full_history_default():
    settings = Settings(
        planner_context_budget_tokens=101,
        answer_context_budget_tokens=202,
        verifier_context_budget_tokens=303,
        qwen_api_key=None,
    )
    catalog = Catalog()
    planner, _ = build_planner(settings, catalog, provider=Provider({"decision": "no_tool", "tasks": []}))
    writer = build_answer_writer(settings, catalog, provider=Provider({"answer": "draft", "evidence": []}))
    verifier = build_verifier(
        settings,
        catalog,
        provider=Provider({"decision": "PASS", "reason": "enough", "missing_evidence": []}),
    )

    assert planner._context_policy.budget_tokens == 101
    assert writer._context_policy.budget_tokens == 202
    assert verifier._context_policy.budget_tokens == 303
    assert planner._context_policy.strategy == "full_history"


def test_replan_and_rewrite_also_apply_selected_history():
    settings = Settings(context_strategy="last_n", context_last_n=1, qwen_api_key=None)
    catalog = Catalog()
    task, result = inputs()
    feedback = VerificationResult(
        decision="REPLAN",
        reason="missing",
        missing_evidence=["new lookup"],
    )
    planner_provider = Provider({
        "tasks": [{
            "task_id": "t1",
            "tool_name": "lookup",
            "arguments": {"value": 7},
            "dependencies": [],
            "bindings": [],
        }],
        "force_rerun_task_ids": [],
    })
    planner, _ = build_planner(settings, catalog, provider=planner_provider)
    planner.replan(request(), [task], [result], feedback)

    writer_provider = Provider({
        "answer": "7",
        "evidence": [{"task_id": "t1", "source_path": ["value"]}],
    })
    writer = build_answer_writer(settings, catalog, provider=writer_provider)
    writer.rewrite(
        request(),
        [task],
        [result],
        DraftAnswer(answer="old"),
        feedback,
    )

    expected = [{"role": "user", "content": "latest"}]
    assert payload(planner_provider)["history"] == expected
    assert payload(writer_provider)["history"] == expected


def test_shared_summary_is_sent_separately_to_all_component_prompts():
    settings = Settings(
        context_strategy="summary_compression",
        context_summary_recent_n=1,
        planner_context_budget_tokens=2,
        answer_context_budget_tokens=2,
        verifier_context_budget_tokens=2,
        qwen_api_key=None,
    )
    summary_provider = Provider({"facts": [{
        "category": "planning_fact", "content": "old", "source_message_index": 0,
    }]})
    manager = build_context_manager(
        UnitEstimator(), settings=settings, summary_provider=summary_provider,
    )
    catalog = Catalog()
    task, result = inputs()
    shared_request = request()

    planner_provider = Provider({"decision": "no_tool", "tasks": []})
    planner, _ = build_planner(settings, catalog, provider=planner_provider, context_manager=manager)
    planner.plan(shared_request)
    writer_provider = Provider({
        "answer": "7", "evidence": [{"task_id": "t1", "source_path": ["value"]}],
    })
    writer = build_answer_writer(settings, catalog, provider=writer_provider, context_manager=manager)
    draft = writer.write(shared_request, [task], [result])
    verifier_provider = Provider({"decision": "PASS", "reason": "enough", "missing_evidence": []})
    verifier = build_verifier(settings, catalog, provider=verifier_provider, context_manager=manager)
    verifier.verify(shared_request, [task], [result], draft)

    for model_provider in (planner_provider, writer_provider, verifier_provider):
        assert payload(model_provider)["history"] == [{"role": "user", "content": "latest"}]
        assert payload(model_provider)["history_summary"]["facts"][0]["source_message_index"] == 0
    assert len(summary_provider.calls) == 1


def test_retrieved_history_is_separate_and_score_is_not_sent_to_components():
    settings = Settings(
        context_strategy="summary_retrieval",
        context_summary_recent_n=1,
        planner_context_budget_tokens=3,
        answer_context_budget_tokens=3,
        verifier_context_budget_tokens=3,
        qwen_api_key=None,
    )
    summary_provider = Provider({"facts": [{
        "category": "planning_fact", "content": "durable preference", "source_message_index": 0,
    }]})
    manager = build_context_manager(UnitEstimator(), settings=settings, summary_provider=summary_provider)
    shared_request = UserQuery(
        query="cash detail",
        history=[
            Message(role="user", content="durable preference"),
            Message(role="assistant", content="noted"),
            Message(role="user", content="cash detail"),
            Message(role="assistant", content="cash answer"),
            Message(role="user", content="latest"),
        ],
    )
    catalog = Catalog()
    task, result = inputs()
    planner_provider = Provider({"decision": "no_tool", "tasks": []})
    planner, _ = build_planner(settings, catalog, provider=planner_provider, context_manager=manager)
    planner.plan(shared_request)
    writer_provider = Provider({
        "answer": "7", "evidence": [{"task_id": "t1", "source_path": ["value"]}],
    })
    writer = build_answer_writer(settings, catalog, provider=writer_provider, context_manager=manager)
    draft = writer.write(shared_request, [task], [result])
    verifier_provider = Provider({"decision": "PASS", "reason": "enough", "missing_evidence": []})
    verifier = build_verifier(settings, catalog, provider=verifier_provider, context_manager=manager)
    verifier.verify(shared_request, [task], [result], draft)

    for model_provider in (planner_provider, writer_provider, verifier_provider):
        body = payload(model_provider)
        assert body["history"] == [{"role": "user", "content": "latest"}]
        assert body["history_summary"]["facts"][0]["content"] == "durable preference"
        assert body["retrieved_history"][0]["message_indexes"] == [2, 3]
        assert "score" not in body["retrieved_history"][0]
    assert len(summary_provider.calls) == 1


def test_runtime_rejects_two_context_extension_points_at_once():
    settings = Settings(qwen_api_key=None)
    provider = Provider({"decision": "no_tool", "tasks": []})
    with pytest.raises(ValueError, match="Pass context_manager or context dependencies, not both"):
        build_planner(
            settings,
            Catalog(),
            provider=provider,
            context_manager=ContextManager(),
            token_estimator=HeuristicTokenEstimator(),
        )
