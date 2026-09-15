from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest

from financial_agent.context import (
    ContextManager,
    ContextPolicy,
    HistorySummarizer,
    QwenSummaryProvider,
    build_context_manager,
)
from financial_agent.config import Settings
from financial_agent.context.qwen_summary_provider import SummaryProviderUnavailableError
from financial_agent.context.summarizer import InvalidHistorySummaryError, ProtectedFactMissingError
from financial_agent.schemas import Message, UserQuery


class Provider:
    def __init__(self, facts):
        self.facts = facts
        self.calls = []

    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return {"facts": self.facts}


class FactCostEstimator:
    def estimate_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for message in messages:
            if message.content.startswith('{"type":"history_summary"'):
                total += len(json.loads(message.content)["facts"])
            else:
                total += 1
        return total


def history():
    return [
        Message(role="user", content="客户是 syn-user-0001"),
        Message(role="assistant", content="好的"),
        Message(role="user", content="需要组合分析"),
        Message(role="assistant", content="可以"),
        Message(role="user", content="最近的问题"),
        Message(role="assistant", content="最近的回答"),
    ]


def policy(budget=3):
    return ContextPolicy(
        strategy="summary_compression",
        budget_tokens=budget,
        summary_recent_n=1,
        summary_budget_ratio=0.4,
    )


def test_negative_summary_cache_size_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ContextManager(summary_cache_size=-1)


def test_summary_preserves_absolute_source_index_and_current_query():
    provider = Provider([{
        "category": "entity", "content": "syn-user-0001", "source_message_index": 0,
    }])
    manager = ContextManager(
        FactCostEstimator(), summarizer=HistorySummarizer(provider),
    )
    request = UserQuery(query="继续分析他", history=history())

    selection = manager.select(request, "planner", policy())

    assert selection.request.query == request.query
    assert [item.content for item in selection.request.history] == ["最近的问题", "最近的回答"]
    assert selection.summary.facts[0].source_message_index == 0
    assert selection.metrics.selected_tokens == 3
    assert selection.metrics.summary_used is True
    assert selection.metrics.summary_fallback_reason is None


def test_complete_grounded_summary_is_cached_but_packed_per_budget():
    provider = Provider([
        {"category": "planning_fact", "content": "需要组合分析", "source_message_index": 2},
        {"category": "entity", "content": "syn-user-0001", "source_message_index": 0},
    ])
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    request = UserQuery(query="继续", history=history())

    small = manager.select(request, "planner", policy(3))
    large = manager.select(request, "writer", policy(4))

    assert len(provider.calls) == 1
    assert small.metrics.summary_cache_hit is False
    assert large.metrics.summary_cache_hit is True
    assert len(small.summary.facts) == 1
    assert len(large.summary.facts) == 2


def test_summary_borrows_unused_raw_budget():
    provider = Provider([
        {"category": "entity", "content": "syn-user-0001", "source_message_index": 0},
        {"category": "planning_fact", "content": "好的", "source_message_index": 1},
        {"category": "planning_fact", "content": "需要组合分析", "source_message_index": 2},
    ])
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    selection = manager.select(UserQuery(query="继续", history=history()), "planner", policy(5))
    assert len(selection.request.history) == 2
    assert len(selection.summary.facts) == 3
    assert selection.metrics.selected_tokens == 5


def test_query_change_does_not_hit_summary_cache():
    facts = [{"category": "entity", "content": "syn-user-0001", "source_message_index": 0}]
    provider = Provider(facts)
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    request = UserQuery(query="first", history=history())
    manager.select(request, "planner", policy())
    manager.select(request.model_copy(update={"query": "second"}), "planner", policy())
    assert len(provider.calls) == 2


def test_request_id_change_does_not_hit_summary_cache():
    facts = [{"category": "entity", "content": "syn-user-0001", "source_message_index": 0}]
    provider = Provider(facts)
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    manager.select(UserQuery(query="same", history=history()), "planner", policy())
    manager.select(UserQuery(query="same", history=history()), "planner", policy())
    assert len(provider.calls) == 2


def test_explicit_summary_strategy_requires_key_only_without_injected_provider():
    settings = Settings(context_strategy="summary_compression", qwen_api_key=None)
    with pytest.raises(ValueError, match="Qwen API key is required for summary_compression"):
        build_context_manager(settings=settings)
    assert build_context_manager(settings=settings, summary_provider=Provider([])) is not None


def test_cache_can_be_cleared_by_request_id():
    provider = Provider([{"category": "entity", "content": "syn-user-0001", "source_message_index": 0}])
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    request = UserQuery(query="continue", history=history())
    manager.select(request, "planner", policy())
    manager.clear_summary_cache(request.request_id)
    manager.select(request, "planner", policy())
    assert len(provider.calls) == 2


def test_history_below_budget_never_calls_summarizer():
    provider = Provider([])
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    request = UserQuery(query="q", history=history()[:2])
    selection = manager.select(request, "planner", policy(2))
    assert selection.request.history == request.history
    assert selection.summary is None
    assert provider.calls == []


def test_missing_protected_fact_falls_back_without_raising():
    provider = Provider([{"category": "planning_fact", "content": "需要组合分析", "source_message_index": 2}])
    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(provider))
    selection = manager.select(UserQuery(query="继续", history=history()), "planner", policy())
    assert selection.summary is None
    assert selection.metrics.summary_fallback_reason == "protected_fact_missing"


def test_grounding_rejects_invalid_source_and_non_source_text():
    indexed = list(enumerate(history()[:4]))
    for fact in (
        {"category": "planning_fact", "content": "不存在", "source_message_index": 2},
        {"category": "planning_fact", "content": "需要组合分析", "source_message_index": 99},
    ):
        with pytest.raises(InvalidHistorySummaryError):
            HistorySummarizer(Provider([fact])).summarize("q", indexed, [])


def test_correction_replaces_old_date_and_must_cite_correction_message():
    indexed = [
        (0, Message(role="user", content="日期是 2026-01-01")),
        (1, Message(role="assistant", content="好的")),
        (2, Message(role="user", content="更正：改为 2026-05-01")),
    ]
    valid = Provider([{
        "category": "time_range", "content": "改为 2026-05-01", "source_message_index": 2,
    }])
    summary = HistorySummarizer(valid).summarize("按更正日期", indexed, [])
    assert summary.facts[0].source_message_index == 2

    stale = Provider([{
        "category": "time_range", "content": "2026-01-01", "source_message_index": 0,
    }])
    with pytest.raises(ProtectedFactMissingError):
        HistorySummarizer(stale).summarize("按更正日期", indexed, [])


def test_recent_raw_correction_supersedes_protected_value_in_summary_prefix():
    prefix = [
        (0, Message(role="user", content="日期是 2026-01-01")),
        (1, Message(role="user", content="需要组合分析")),
    ]
    recent = [(2, Message(role="user", content="更正：改为 2026-05-01"))]
    provider = Provider([{
        "category": "planning_fact", "content": "需要组合分析", "source_message_index": 1,
    }])

    summary = HistorySummarizer(provider).summarize("继续", prefix, recent)

    assert [fact.content for fact in summary.facts] == ["需要组合分析"]


def test_provider_error_falls_back_and_is_classified():
    class FailingProvider:
        def generate(self, messages, *, response_schema):
            raise SummaryProviderUnavailableError("offline")

    manager = ContextManager(FactCostEstimator(), summarizer=HistorySummarizer(FailingProvider()))
    selection = manager.select(UserQuery(query="q", history=history()), "planner", policy())
    assert selection.summary is None
    assert selection.metrics.summary_fallback_reason == "provider_error"
    assert selection.metrics.summarized_message_count == 4


def test_qwen_summary_provider_uses_strict_schema_and_redacts_key():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"facts":[]}'}}]})

    provider = QwenSummaryProvider(
        "test-summary-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.generate([], response_schema={"type": "object"}) == {"facts": []}
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
    assert captured["body"]["enable_thinking"] is False
    assert "test-summary-secret" not in repr(provider)
