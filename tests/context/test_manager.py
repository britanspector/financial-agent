from __future__ import annotations

import logging
from collections.abc import Sequence

from financial_agent.context import (
    ContextManager,
    ContextPolicy,
    HeuristicTokenEstimator,
)
from financial_agent.schemas import Message, UserQuery


def messages(*items: tuple[str, str]) -> list[Message]:
    return [Message(role=role, content=content) for role, content in items]


class ContentCostEstimator:
    def estimate_messages(self, history: Sequence[Message]) -> int:
        return sum(10 if "OVERSIZED" in item.content else 1 for item in history)


def select(request, policy, *, estimator=None, component="planner"):
    return ContextManager(estimator).select(request, component, policy)


def test_full_history_preserves_request_and_reports_uncompressed_metrics():
    history = messages(("user", "first"), ("assistant", "answer"))
    request = UserQuery(query="current", history=history)

    selection = select(request, ContextPolicy(strategy="full_history", budget_tokens=0))

    assert selection.request.request_id == request.request_id
    assert selection.request.query == "current"
    assert selection.request.history == history
    assert request.history == history
    assert selection.request is not request
    assert selection.metrics.selected_tokens == selection.metrics.original_tokens
    assert selection.metrics.compression_ratio == 1
    assert selection.metrics.selected_message_count == 2
    assert selection.metrics.dropped_message_count == 0


def test_empty_history_has_zero_tokens_and_unit_compression_ratio():
    selection = select(
        UserQuery(query="current"),
        ContextPolicy(strategy="budgeted_selection", budget_tokens=0),
    )
    assert selection.request.history == []
    assert selection.metrics.original_tokens == 0
    assert selection.metrics.selected_tokens == 0
    assert selection.metrics.compression_ratio == 1


def test_last_n_groups_user_turns_and_leading_assistant_messages():
    history = messages(
        ("assistant", "orphan one"),
        ("assistant", "orphan two"),
        ("user", "first user"),
        ("assistant", "first answer"),
        ("user", "second user"),
        ("user", "third user"),
        ("assistant", "third answer"),
    )
    request = UserQuery(query="current", history=history)

    last_two = select(request, ContextPolicy(strategy="last_n", last_n=2))
    none = select(request, ContextPolicy(strategy="last_n", last_n=0))

    assert [item.content for item in last_two.request.history] == [
        "second user", "third user", "third answer",
    ]
    assert none.request.history == []
    assert none.request.query == "current"


def test_budgeted_selection_skips_oversized_latest_turn_and_keeps_relevant_history():
    request = UserQuery(
        query="华泰科技后续怎么看",
        history=messages(
            ("user", "请分析华泰科技"),
            ("user", "今天天气如何"),
            ("user", "OVERSIZED latest turn"),
        ),
    )

    selection = select(
        request,
        ContextPolicy(strategy="budgeted_selection", budget_tokens=1),
        estimator=ContentCostEstimator(),
    )

    assert [item.content for item in selection.request.history] == ["请分析华泰科技"]
    assert selection.metrics.selected_tokens == 1
    assert selection.request.query == request.query


def test_budgeted_selection_prefers_shared_deterministic_entities():
    request = UserQuery(
        query="比较 600519.SH 在 2026-09-01 的情况",
        history=messages(
            ("user", "此前讨论 600519.SH，日期是 2026-09-01"),
            ("user", "一个较新的无关话题"),
            ("user", "OVERSIZED latest turn"),
        ),
    )
    selection = select(
        request,
        ContextPolicy(strategy="budgeted_selection", budget_tokens=1),
        estimator=ContentCostEstimator(),
    )
    assert selection.request.history[0].content.startswith("此前讨论 600519.SH")


def test_user_constraint_beats_plain_assistant_acknowledgement():
    request = UserQuery(
        query="继续处理",
        history=messages(
            ("user", "必须只使用历史收盘价，不要预测"),
            ("user", "普通上下文"),
            ("assistant", "收到，已确认"),
            ("user", "OVERSIZED latest turn"),
        ),
    )
    selection = select(
        request,
        ContextPolicy(strategy="budgeted_selection", budget_tokens=1),
        estimator=ContentCostEstimator(),
    )
    assert [item.content for item in selection.request.history] == ["必须只使用历史收盘价，不要预测"]


def test_company_name_is_selected_through_lexical_overlap():
    request = UserQuery(
        query="华泰科技的研报观点",
        history=messages(
            ("user", "华泰科技盈利趋势"),
            ("user", "另一个无关问题"),
            ("user", "OVERSIZED latest turn"),
        ),
    )
    selection = select(
        request,
        ContextPolicy(strategy="budgeted_selection", budget_tokens=1),
        estimator=ContentCostEstimator(),
    )
    assert [item.content for item in selection.request.history] == ["华泰科技盈利趋势"]


def test_selection_is_deterministic_and_preserves_original_order():
    request = UserQuery(
        query="syn-user-0001 600519.SH",
        history=messages(
            ("user", "syn-user-0001 的持仓"),
            ("assistant", "持仓已查询"),
            ("user", "600519.SH 的行情"),
        ),
    )
    policy = ContextPolicy(strategy="budgeted_selection", budget_tokens=3)
    manager = ContextManager(ContentCostEstimator())

    first = manager.select(request, "writer", policy)
    second = manager.select(request, "writer", policy)

    assert first == second
    assert first.request.history == request.history


def test_safe_metrics_log_contains_no_request_text_or_entity(caplog):
    secret_text = "never-log-this syn-user-0001"
    request = UserQuery(query="also-private", history=messages(("user", secret_text)))
    with caplog.at_level(logging.INFO, logger="financial_agent.context.manager"):
        select(request, ContextPolicy(strategy="last_n", last_n=0), component="verifier")

    output = caplog.text
    assert "context_selection component=verifier" in output
    assert "selected_message_count=0" in output
    assert secret_text not in output
    assert "also-private" not in output
    assert "syn-user-0001" not in output


def test_heuristic_estimator_is_stable_and_counts_no_empty_envelope():
    estimator = HeuristicTokenEstimator()
    history = messages(("user", "中文 and ASCII"))
    assert estimator.estimate_messages([]) == 0
    assert estimator.estimate_messages(history) == estimator.estimate_messages(history)
    assert estimator.estimate_messages(history) > 0
