import json
from uuid import uuid4

import httpx
import pytest
from pydantic import RootModel

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.answering.providers import AnswerProviderResponseError
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.answering.service import AnswerWriter
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.contracts import ToolResult


class Output(Schema):
    value: int


class ListItem(Schema):
    value: int


class ListOutput(Schema):
    items: list[ListItem]


class SearchItem(Schema):
    document_id: str
    title: str
    content: str


class SearchOutput(RootModel[list[SearchItem]]):
    pass


class Catalog:
    def output_model(self, name): return Output


class ListCatalog:
    def output_model(self, name): return ListOutput


class SearchCatalog:
    def output_model(self, name): return SearchOutput


class Provider:
    def __init__(self, output): self.output, self.calls = output, []
    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema)); return self.output


def inputs():
    task = Task(task_id="a", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(task_id="a", tool_name="lookup",
        result=ToolResult(status="success", data={"value": 7}, source="test", latency=0,
                          error=None, request_id=uuid4()))
    return task, result


def test_writer_returns_validated_evidence_grounded_draft():
    task, result = inputs()
    provider = Provider({"answer": "7", "evidence": [{"task_id": "a", "source_path": ["value"]}]})
    draft = AnswerWriter(provider, Catalog()).write(UserQuery(query="q"), [task], [result])
    payload = json.loads(provider.calls[0][0][1]["content"])
    assert draft.answer == "7"
    assert 'cite a data field named value as ["value"]' in provider.calls[0][0][0]["content"]
    assert "request_id" not in provider.calls[0][0][1]["content"]
    assert payload["tool_results"][0]["data"] == {"value": 7}


def test_writer_rejects_non_public_evidence_path():
    task, result = inputs()
    provider = Provider({"answer": "7", "evidence": [{"task_id": "a", "source_path": ["secret"]}]})
    with pytest.raises(AnswerProviderResponseError):
        AnswerWriter(provider, Catalog()).write(UserQuery(query="q"), [task], [result])


def test_writer_normalizes_redundant_tool_result_data_prefix():
    task, result = inputs()
    provider = Provider({
        "answer": "7",
        "evidence": [{"task_id": "a", "source_path": ["data", "value"]}],
    })

    draft = AnswerWriter(provider, Catalog()).write(UserQuery(query="q"), [task], [result])

    assert draft.evidence[0].source_path == ["value"]


def test_writer_normalizes_bracketed_list_index_in_evidence_path():
    task = Task(task_id="a", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="a",
        tool_name="lookup",
        result=ToolResult(
            status="success",
            data={"items": [{"value": 7}]},
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "7",
        "evidence": [{"task_id": "a", "source_path": ["data", "items", "[0]", "value"]}],
    })

    draft = AnswerWriter(provider, ListCatalog()).write(UserQuery(query="q"), [task], [result])

    assert draft.evidence[0].source_path == ["items", 0, "value"]


def test_writer_normalizes_serialized_result_object_in_evidence_path():
    task = Task(task_id="a", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="a",
        tool_name="lookup",
        result=ToolResult(
            status="success",
            data={"items": [{"value": 7}, {"value": 9}]},
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "7",
        "evidence": [{
            "task_id": "a",
            "source_path": ["data", json.dumps({"value": 7})],
        }],
    })

    draft = AnswerWriter(provider, ListCatalog()).write(
        UserQuery(query="q"), [task], [result],
    )

    assert draft.evidence[0].source_path == ["items", 0]


def test_writer_does_not_guess_ambiguous_serialized_result_object_path():
    task = Task(task_id="a", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="a",
        tool_name="lookup",
        result=ToolResult(
            status="success",
            data={"items": [{"value": 7}, {"value": 7}]},
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "7",
        "evidence": [{
            "task_id": "a",
            "source_path": ["data", json.dumps({"value": 7})],
        }],
    })

    with pytest.raises(AnswerProviderResponseError):
        AnswerWriter(provider, ListCatalog()).write(
            UserQuery(query="q"), [task], [result],
        )


def test_writer_normalizes_compact_evidence_path():
    task = Task(task_id="a", tool_name="lookup", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="a",
        tool_name="lookup",
        result=ToolResult(
            status="success",
            data={"items": [{"value": 7}]},
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "7",
        "evidence": [{"task_id": "a", "source_path": ["data.items[0].value"]}],
    })

    draft = AnswerWriter(provider, ListCatalog()).write(UserQuery(query="q"), [task], [result])

    assert draft.evidence[0].source_path == ["items", 0, "value"]


def test_writer_adds_index_for_field_on_list_root():
    task = Task(task_id="a", tool_name="search", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="a",
        tool_name="search",
        result=ToolResult(
            status="success",
            data=[{
                "document_id": "doc-1",
                "title": "Margin rules",
                "content": "Maintenance ratio definition",
            }],
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "definition",
        "evidence": [
            {"task_id": "a", "source_path": ["title"]},
            {"task_id": "a", "source_path": ["content"]},
        ],
    })

    draft = AnswerWriter(provider, SearchCatalog()).write(UserQuery(query="q"), [task], [result])

    assert [reference.source_path for reference in draft.evidence] == [
        [0, "title"],
        [0, "content"],
    ]


def test_writer_maps_document_id_to_task_and_result_index():
    task = Task(task_id="search-1", tool_name="search", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="search-1",
        tool_name="search",
        result=ToolResult(
            status="success",
            data=[
                SearchItem(document_id="doc-1", title="First", content="First content"),
                SearchItem(document_id="doc-2", title="Second", content="Second content"),
            ],
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "definition",
        "evidence": [{"task_id": "doc-2", "source_path": ["content"]}],
    })

    draft = AnswerWriter(provider, SearchCatalog()).write(UserQuery(query="q"), [task], [result])

    assert draft.evidence[0].task_id == "search-1"
    assert draft.evidence[0].source_path == [1, "content"]


def test_writer_maps_data_envelope_to_top_search_content():
    task = Task(task_id="search-1", tool_name="search", arguments={}, dependencies=[])
    result = TaskExecutionResult(
        task_id="search-1",
        tool_name="search",
        result=ToolResult(
            status="success",
            data=[SearchItem(
                document_id="doc-1",
                title="Margin rules",
                content="Maintenance ratio definition",
            )],
            source="test",
            latency=0,
            error=None,
            request_id=uuid4(),
        ),
    )
    provider = Provider({
        "answer": "definition",
        "evidence": [{"task_id": "search-1", "source_path": ["data"]}],
    })

    draft = AnswerWriter(provider, SearchCatalog()).write(UserQuery(query="q"), [task], [result])

    assert draft.evidence[0].source_path == [0, "content"]


def test_qwen_answer_provider_uses_strict_structured_output():
    captured = {}
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"ok","evidence":[]}'}}]})
    provider = QwenAnswerProvider("secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.generate([], response_schema={"type": "object"})["answer"] == "ok"
    assert captured["body"]["model"] == "qwen3.7-flash"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
    assert provider.last_raw_response == {"answer": "ok", "evidence": []}
