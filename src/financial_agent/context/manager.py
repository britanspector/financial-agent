"""Deterministic, budget-aware history context management."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from financial_agent.context.models import (
    ContextComponent,
    ContextMetrics,
    ContextPolicy,
    ContextSelection,
)
from financial_agent.context.token_estimation import HeuristicTokenEstimator, TokenEstimator
from financial_agent.schemas import Message, UserQuery


logger = logging.getLogger(__name__)

_TERM = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.IGNORECASE)
_USER_ID = re.compile(
    r"\bsyn-user-\d{4}\b|\buser[_-]?id\s*[:=：]?\s*[a-z0-9._-]+\b",
    re.IGNORECASE,
)
_SYMBOL = re.compile(r"(?<!\d)\d{6}(?:\.(?:sh|sz))?(?!\d)", re.IGNORECASE)
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)")
_ZH_DATE = re.compile(r"(?<!\d)\d{4}年\d{1,2}月\d{1,2}日")
_USER_IMPORTANCE = re.compile(
    r"必须|务必|只要|仅|不要|不得|不能|优先|截至|先.+再|改为|不是|更正|纠正|确认|没错|对的|"
    r"\bmust\b|\bonly\b|\bdo\s+not\b|\bdon't\b|\binstead\b|\bcorrect(?:ion|ed)?\b|"
    r"\bconfirm(?:ed)?\b",
    re.IGNORECASE,
)


class ContextManager:
    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self._estimator = estimator or HeuristicTokenEstimator()

    def select(
        self,
        request: UserQuery,
        component: ContextComponent,
        policy: ContextPolicy,
    ) -> ContextSelection:
        original = list(request.history)
        if policy.strategy == "full_history":
            selected = original
        elif policy.strategy == "last_n":
            turns = _group_turns(original)
            selected = _flatten(turns[-policy.last_n:]) if policy.last_n else []
        else:
            selected = self._budgeted_selection(request.query, original, policy.budget_tokens)

        original_tokens = self._estimator.estimate_messages(original)
        selected_tokens = self._estimator.estimate_messages(selected)
        ratio = selected_tokens / original_tokens if original_tokens else 1.0
        metrics = ContextMetrics(
            component=component,
            strategy=policy.strategy,
            budget_tokens=policy.budget_tokens,
            original_tokens=original_tokens,
            selected_tokens=selected_tokens,
            compression_ratio=ratio,
            selected_message_count=len(selected),
            dropped_message_count=len(original) - len(selected),
        )
        logger.info(
            "context_selection component=%s strategy=%s budget_tokens=%d original_tokens=%d "
            "selected_tokens=%d compression_ratio=%.6f selected_message_count=%d dropped_message_count=%d",
            metrics.component,
            metrics.strategy,
            metrics.budget_tokens,
            metrics.original_tokens,
            metrics.selected_tokens,
            metrics.compression_ratio,
            metrics.selected_message_count,
            metrics.dropped_message_count,
        )
        return ContextSelection(
            request=UserQuery(
                request_id=request.request_id,
                query=request.query,
                history=selected,
            ),
            metrics=metrics,
        )

    def _budgeted_selection(
        self,
        query: str,
        history: Sequence[Message],
        budget_tokens: int,
    ) -> list[Message]:
        turns = _group_turns(history)
        if not turns or budget_tokens == 0:
            return []

        query_terms = _terms(query)
        query_entities = _entities(query)
        latest_index = len(turns) - 1
        remaining = list(range(latest_index))
        remaining.sort(
            key=lambda index: (
                _turn_score(
                    turns[index],
                    age=latest_index - index,
                    query_terms=query_terms,
                    query_entities=query_entities,
                ),
                index,
            ),
            reverse=True,
        )

        chosen: list[int] = []
        for index in [latest_index, *remaining]:
            candidate_indexes = sorted([*chosen, index])
            candidate = _flatten([turns[item] for item in candidate_indexes])
            if self._estimator.estimate_messages(candidate) <= budget_tokens:
                chosen.append(index)
        return _flatten([turns[index] for index in sorted(chosen)])


def _group_turns(history: Sequence[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    current: list[Message] = []
    for message in history:
        if message.role == "user":
            if current:
                turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _flatten(turns: Sequence[Sequence[Message]]) -> list[Message]:
    return [message for turn in turns for message in turn]


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _TERM.finditer(text.lower()):
        value = match.group(0)
        if "\u4e00" <= value[0] <= "\u9fff":
            if len(value) == 1:
                terms.add(value)
            else:
                terms.update(value[index:index + 2] for index in range(len(value) - 1))
        else:
            terms.add(value)
    return terms


def _entities(text: str) -> set[str]:
    entities: set[str] = set()
    for kind, pattern in (
        ("user_id", _USER_ID),
        ("symbol", _SYMBOL),
        ("date", _ISO_DATE),
        ("date", _ZH_DATE),
    ):
        entities.update(f"{kind}:{match.group(0).lower()}" for match in pattern.finditer(text))
    return entities


def _turn_score(
    turn: Sequence[Message],
    *,
    age: int,
    query_terms: set[str],
    query_entities: set[str],
) -> float:
    text = "\n".join(message.content for message in turn)
    turn_terms = _terms(text)
    turn_entities = _entities(text)
    relevance = len(query_terms & turn_terms) / max(1, len(query_terms))
    shared_entities = len(query_entities & turn_entities)
    user_important = any(
        message.role == "user" and _USER_IMPORTANCE.search(message.content)
        for message in turn
    )
    return (
        100 / (age + 1)
        + 60 * relevance
        + 40 * min(shared_entities, 3)
        + (30 if turn_entities else 0)
        + (30 if user_important else 0)
    )
