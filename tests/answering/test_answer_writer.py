import json
from uuid import uuid4

import httpx
import pytest

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.answering.providers import AnswerProviderResponseError
from financial_agent.answering.qwen_provider import QwenAnswerProvider
from financial_agent.answering.service import AnswerWriter
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.contracts import ToolResult


class Output(Schema):
    value: int


class Catalog:
    def output_model(self, name): return Output


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


def test_qwen_answer_provider_uses_strict_structured_output():
    captured = {}
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"ok","evidence":[]}'}}]})
    provider = QwenAnswerProvider("secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.generate([], response_schema={"type": "object"})["answer"] == "ok"
    assert captured["body"]["model"] == "qwen3.7-flash"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
