from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from financial_agent.config import Settings
from financial_agent.context import (
    ContextManager,
    ContextPolicy,
    HistorySummarizer,
    HistorySummary,
    SummaryFact,
    build_context_manager,
)
from financial_agent.context.retrieval import HistoryTurn, LexicalHistoryRetriever
from financial_agent.context.summary_providers import SummaryProviderUnavailableError
from financial_agent.schemas import Message, UserQuery


class Provider:
    def __init__(self, facts):
        self.facts = facts
        self.calls = []

    def generate(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return {"facts": self.facts}


class UnitEstimator:
    def estimate_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for message in messages:
            if message.content.startswith('{"type":"history_summary"'):
                total += len(json.loads(message.content)["facts"])
            elif message.content.startswith('{"type":"retrieved_history"'):
                total += sum(len(turn["messages"]) for turn in json.loads(message.content)["turns"])
            else:
                total += 1
        return total


def turn(index, start, user, assistant="收到"):
    return HistoryTurn(
        turn_index=index,
        message_indexes=(start, start + 1),
        messages=(Message(role="user", content=user), Message(role="assistant", content=assistant)),
    )


def test_lexical_bm25_ranks_chinese_match_and_uses_recency_as_tie_break():
    retriever = LexicalHistoryRetriever()
    candidates = [
        turn(0, 0, "讨论现金流风险"),
        turn(1, 2, "讨论现金流风险"),
        turn(2, 4, "无关的报告格式"),
    ]
    history = [message for item in candidates for message in item.messages]

    result = retriever.retrieve(
        "现金流风险", candidates, stable_summary=None, full_history=history, top_k=4, min_score=0.15,
    )

    assert [item.turn_index for item in result.hits] == [1, 0]
    assert result.candidate_count == 3
    assert result.eligible_count == 2


def test_entity_boost_and_query_expansion_recover_referenced_company():
    retriever = LexicalHistoryRetriever()
    candidates = [turn(0, 0, "华泰科技现金流承压"), turn(1, 2, "另一家公司现金流正常")]
    summary = HistorySummary(facts=[SummaryFact(
        category="entity", content="华泰科技", source_message_index=0,
    )])

    result = retriever.retrieve(
        "继续分析它", candidates, stable_summary=summary,
        full_history=[message for item in candidates for message in item.messages],
        top_k=1, min_score=0.15,
    )

    assert result.hits[0].turn_index == 0
    assert result.hits[0].score >= 0.25


def test_threshold_can_leave_retrieval_inactive():
    candidate = turn(0, 0, "组合风险分析")
    result = LexicalHistoryRetriever().retrieve(
        "天气", [candidate], stable_summary=None, full_history=list(candidate.messages),
        top_k=4, min_score=0.15,
    )
    assert result.hits == ()
    assert result.eligible_count == 0


def test_corrected_old_date_turn_is_filtered():
    old = turn(0, 0, "范围是 2026-01-01")
    corrected = turn(1, 2, "更正：改为 2026-05-01")
    history = [message for item in (old, corrected) for message in item.messages]
    result = LexicalHistoryRetriever().retrieve(
        "2026-01-01", [old], stable_summary=None, full_history=history,
        top_k=4, min_score=0.15,
    )
    assert result.hits == ()


def test_summary_retrieval_packs_recent_retrieved_and_protected_summary_without_duplication():
    messages = [
        Message(role="user", content="客户是 syn-user-0001"),
        Message(role="assistant", content="已记录"),
        Message(role="user", content="回答保持简洁"),
        Message(role="assistant", content="好的"),
        Message(role="user", content="华泰科技现金流风险需要重点分析"),
        Message(role="assistant", content="现金流存在压力"),
        Message(role="user", content="最近聊一下输出格式"),
        Message(role="assistant", content="可以分段"),
    ]
    provider = Provider([
        {"category": "entity", "content": "syn-user-0001", "source_message_index": 0},
        {"category": "planning_fact", "content": "回答保持简洁", "source_message_index": 2},
        {"category": "planning_fact", "content": "现金流存在压力", "source_message_index": 5},
    ])
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider))
    policy = ContextPolicy(
        strategy="summary_retrieval", budget_tokens=5, summary_recent_n=1,
        retrieval_recent_reservation_ratio=0.3,
        retrieval_protected_summary_reservation_ratio=0.1,
        retrieval_history_reservation_ratio=0.1,
    )

    selection = manager.select(UserQuery(query="继续分析现金流风险", history=messages), "planner", policy)

    assert selection.metrics.selected_tokens <= 5
    assert selection.metrics.retrieval_active
    assert [item.content for item in selection.request.history] == ["最近聊一下输出格式", "可以分段"]
    assert selection.retrieved_history[0].message_indexes == [4, 5]
    assert [fact.content for fact in selection.summary.facts] == ["syn-user-0001"]
    assert selection.metrics.protected_fact_count == 1
    assert selection.metrics.protected_fact_covered_count == 1


def test_same_stable_prefix_is_cached_across_query_changes():
    messages = [
        Message(role="user", content="客户是 syn-user-0001"),
        Message(role="assistant", content="已记录"),
        Message(role="user", content="最近问题"),
        Message(role="assistant", content="最近回答"),
    ]
    provider = Provider([{
        "category": "entity", "content": "syn-user-0001", "source_message_index": 0,
    }])
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider))
    policy = ContextPolicy(strategy="summary_retrieval", budget_tokens=3, summary_recent_n=1)
    request = UserQuery(query="first", history=messages)

    first = manager.select(request, "planner", policy)
    second = manager.select(request.model_copy(update={"query": "second"}), "writer", policy)

    assert len(provider.calls) == 1
    assert not first.metrics.summary_cache_hit
    assert second.metrics.summary_cache_hit


def test_inactive_retrieval_uses_summary_and_returns_retrieval_reservation_to_pool():
    messages = [
        Message(role="user", content="durable preference"),
        Message(role="assistant", content="noted"),
        Message(role="user", content="latest"),
    ]
    provider = Provider([{
        "category": "planning_fact", "content": "durable preference", "source_message_index": 0,
    }])
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider))
    selection = manager.select(
        UserQuery(query="unrelated weather", history=messages), "planner",
        ContextPolicy(strategy="summary_retrieval", budget_tokens=2, summary_recent_n=1),
    )
    assert not selection.metrics.retrieval_active
    assert selection.metrics.retrieved_tokens == 0
    assert selection.summary.facts[0].content == "durable preference"
    assert selection.request.history[0].content == "latest"


def test_retriever_failure_degrades_to_summary_and_recent():
    class FailingRetriever:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("local index failed")

    messages = [
        Message(role="user", content="durable preference"),
        Message(role="assistant", content="noted"),
        Message(role="user", content="latest"),
    ]
    manager = ContextManager(
        UnitEstimator(),
        summarizer=HistorySummarizer(Provider([{
            "category": "planning_fact", "content": "durable preference", "source_message_index": 0,
        }])),
        retriever=FailingRetriever(),
    )
    selection = manager.select(
        UserQuery(query="durable", history=messages), "planner",
        ContextPolicy(strategy="summary_retrieval", budget_tokens=2, summary_recent_n=1),
    )
    assert selection.summary is not None
    assert selection.retrieved_history == []
    assert selection.metrics.retrieval_fallback_reason == "retriever_error"


def test_summary_failure_degrades_to_retrieved_and_recent():
    class FailingProvider:
        def generate(self, messages, *, response_schema):
            raise SummaryProviderUnavailableError("offline")

    messages = [
        Message(role="user", content="cash risk"),
        Message(role="assistant", content="cash detail"),
        Message(role="user", content="format note"),
        Message(role="assistant", content="noted"),
        Message(role="user", content="latest"),
    ]
    manager = ContextManager(
        UnitEstimator(), summarizer=HistorySummarizer(FailingProvider()),
    )
    selection = manager.select(
        UserQuery(query="cash risk", history=messages), "planner",
        ContextPolicy(strategy="summary_retrieval", budget_tokens=3, summary_recent_n=1),
    )
    assert selection.summary is None
    assert selection.retrieved_history[0].message_indexes == [0, 1]
    assert selection.metrics.retrieval_fallback_reason == "summary_error"
    assert selection.metrics.summary_fallback_reason == "provider_error"


def test_retrieved_raw_suppresses_duplicate_protected_summary_fact():
    messages = [
        Message(role="user", content="客户是 syn-user-0001"),
        Message(role="assistant", content="已记录"),
        Message(role="user", content="format note"),
        Message(role="assistant", content="noted"),
        Message(role="user", content="latest"),
    ]
    provider = Provider([{
        "category": "entity", "content": "syn-user-0001", "source_message_index": 0,
    }])
    selection = ContextManager(
        UnitEstimator(), summarizer=HistorySummarizer(provider),
    ).select(
        UserQuery(query="分析 syn-user-0001", history=messages), "planner",
        ContextPolicy(strategy="summary_retrieval", budget_tokens=3, summary_recent_n=1),
    )
    assert selection.retrieved_history[0].message_indexes == [0, 1]
    assert selection.summary is None
    assert selection.metrics.protected_fact_covered_count == 1


def test_protected_fact_over_budget_falls_back_without_exceeding_budget():
    messages = [
        Message(role="user", content="客户是 syn-user-0001"),
        Message(role="assistant", content="已记录"),
        Message(role="user", content="最近问题"),
    ]
    provider = Provider([{
        "category": "entity", "content": "syn-user-0001", "source_message_index": 0,
    }])
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider))
    selection = manager.select(
        UserQuery(query="继续", history=messages), "planner",
        ContextPolicy(strategy="summary_retrieval", budget_tokens=0, summary_recent_n=1),
    )
    assert selection.metrics.selected_tokens == 0
    assert selection.metrics.retrieval_fallback_reason == "protected_fact_over_budget"


def test_summary_retrieval_requires_qwen_key_only_without_injected_summary_provider():
    settings = Settings(context_strategy="summary_retrieval", qwen_api_key=None)
    with pytest.raises(ValueError, match="required for summary_retrieval"):
        build_context_manager(settings=settings)
    assert build_context_manager(settings=settings, summary_provider=Provider([])) is not None
