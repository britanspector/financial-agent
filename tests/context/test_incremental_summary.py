from __future__ import annotations

import json
from collections.abc import Sequence

from financial_agent.context import ContextManager, ContextPolicy, HistorySummarizer
from financial_agent.schemas import Message, UserQuery


class UnitEstimator:
    def estimate_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for message in messages:
            if message.content.startswith('{"type":"history_summary"'):
                total += len(json.loads(message.content)["facts"])
            else:
                total += 1
        return total


class IncrementalProvider:
    def __init__(self, *, invalid_update: bool = False) -> None:
        self.calls = []
        self.invalid_update = invalid_update

    def generate(self, messages, *, response_schema):
        del response_schema
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        if "newly_aged_out_history" in payload:
            if self.invalid_update:
                return {"invalid": True}
            additions = []
            replacements = []
            for item in payload["newly_aged_out_history"]:
                if "改为" in item["content"]:
                    replacements.append({
                        "existing_fact_index": 0,
                        "fact": {
                            "category": "time_range",
                            "content": item["content"],
                            "source_message_index": item["index"],
                        },
                    })
                elif item["role"] == "user" and item["content"] != "闲聊":
                    additions.append({
                        "category": "planning_fact",
                        "content": item["content"],
                        "source_message_index": item["index"],
                    })
            return {"additions": additions, "replacements": replacements}
        facts = []
        for item in payload["summarized_history"]:
            if item["role"] != "user" or item["content"] == "闲聊":
                continue
            category = "time_range" if "日期" in item["content"] or "改为" in item["content"] else "entity"
            facts.append({
                "category": category,
                "content": item["content"],
                "source_message_index": item["index"],
            })
        return {"facts": facts}


def policy(**updates):
    values = {
        "strategy": "summary_compression",
        "budget_tokens": 3,
        "summary_recent_n": 1,
        "summary_budget_ratio": 0.4,
    }
    values.update(updates)
    return ContextPolicy(**values)


def messages(*contents):
    result = []
    for content in contents:
        result.extend([
            Message(role="user", content=content),
            Message(role="assistant", content="收到"),
        ])
    return result


def test_longest_cached_prefix_is_incrementally_updated_and_exact_hit_is_free():
    provider = IncrementalProvider()
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider, max_facts=8))

    first = manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0042", "最近问题")),
        "planner", policy(),
    )
    second_request = UserQuery(
        query="继续", history=messages("客户 syn-user-0042", "分析现金流", "最近问题"),
    )
    second = manager.select(second_request, "planner", policy(budget_tokens=4))
    third = manager.select(second_request, "writer", policy(budget_tokens=4))

    assert first.metrics.summary_rebuild_count == 1
    assert first.metrics.summary_provider_call_count == 1
    assert first.metrics.summary_prefix_stability_ratio == 0
    assert second.metrics.summary_incremental_update_count == 1
    assert second.metrics.summary_rebuild_count == 0
    assert second.metrics.summary_provider_call_count == 1
    assert second.metrics.summary_prefix_stability_ratio is not None
    assert second.metrics.summary_prefix_stability_ratio > 0
    assert [fact.source_message_index for fact in second.summary.facts] == [0, 2]
    assert third.metrics.summary_cache_hit
    assert third.metrics.summary_provider_call_count == 0
    assert third.metrics.summary_prefix_stability_ratio == 1
    incremental_payload = provider.calls[1]
    assert [item["index"] for item in incremental_payload["newly_aged_out_history"]] == [2, 3]
    assert "summarized_history" not in incremental_payload


def test_invalid_incremental_update_falls_back_to_full_prefix_rebuild():
    provider = IncrementalProvider(invalid_update=True)
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider, max_facts=8))
    manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0042", "最近问题")),
        "planner", policy(),
    )

    selection = manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0042", "分析现金流", "最近问题")),
        "planner", policy(),
    )

    assert selection.summary is not None
    assert selection.metrics.summary_incremental_update_count == 1
    assert selection.metrics.summary_rebuild_count == 1
    assert selection.metrics.summary_provider_call_count == 2
    assert "newly_aged_out_history" in provider.calls[1]
    assert len(provider.calls[2]["summarized_history"]) == 4


def test_incremental_replacement_grounds_correction_and_removes_old_value():
    provider = IncrementalProvider()
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider, max_facts=8))
    manager.select(
        UserQuery(query="继续", history=messages("日期是 2026-01-01", "最近问题")),
        "planner", policy(),
    )

    selection = manager.select(
        UserQuery(
            query="继续",
            history=messages("日期是 2026-01-01", "更正：改为 2026-05-01", "最近问题"),
        ),
        "planner", policy(),
    )

    assert [fact.source_message_index for fact in selection.summary.facts] == [2]
    assert "2026-05-01" in selection.summary.facts[0].content
    assert all("2026-01-01" not in fact.content for fact in selection.summary.facts)


def test_branch_history_and_disabled_incremental_rebuild_instead_of_updating():
    provider = IncrementalProvider()
    manager = ContextManager(UnitEstimator(), summarizer=HistorySummarizer(provider, max_facts=8))
    manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0042", "最近问题")),
        "planner", policy(),
    )
    branched = manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0099", "最近问题")),
        "planner", policy(),
    )
    extended = manager.select(
        UserQuery(query="继续", history=messages("客户 syn-user-0099", "分析现金流", "最近问题")),
        "planner", policy(summary_incremental_enabled=False),
    )

    assert branched.metrics.summary_rebuild_count == 1
    assert branched.metrics.summary_incremental_update_count == 0
    assert extended.metrics.summary_rebuild_count == 1
    assert extended.metrics.summary_incremental_update_count == 0
