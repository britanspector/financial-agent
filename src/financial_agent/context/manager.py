"""Deterministic, budget-aware history context management."""

from __future__ import annotations

import logging
import re
import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from threading import Lock

from financial_agent.context.models import (
    ContextComponent,
    ContextMetrics,
    ContextPolicy,
    ContextSelection,
    HistorySummary,
    SummaryFact,
)
from financial_agent.context.summarizer import (
    HistorySummarizer,
    InvalidHistorySummaryError,
    ProtectedFact,
    ProtectedFactMissingError,
    extract_protected_facts,
)
from financial_agent.context.summary_prompt import SUMMARY_SCHEMA_VERSION
from financial_agent.context.summary_providers import SummaryProviderError
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
    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        *,
        summarizer: HistorySummarizer | None = None,
        summary_cache_size: int = 128,
    ) -> None:
        if summary_cache_size < 0:
            raise ValueError("summary_cache_size must be non-negative")
        self._estimator = estimator or HeuristicTokenEstimator()
        self._summarizer = summarizer
        self._summary_cache_size = summary_cache_size
        self._summary_cache: OrderedDict[str, tuple[str, HistorySummary]] = OrderedDict()
        self._cache_lock = Lock()

    def select(
        self,
        request: UserQuery,
        component: ContextComponent,
        policy: ContextPolicy,
    ) -> ContextSelection:
        original = list(request.history)
        summary = None
        cache_hit = False
        summarized_count = 0
        fallback_reason = None
        if policy.strategy == "full_history":
            selected = original
        elif policy.strategy == "last_n":
            turns = _group_turns(original)
            selected = _flatten(turns[-policy.last_n:]) if policy.last_n else []
        elif policy.strategy == "budgeted_selection":
            selected = self._budgeted_selection(request.query, original, policy.budget_tokens)
        else:
            selected, summary, cache_hit, summarized_count, fallback_reason = self._summary_compression(
                request, policy,
            )

        original_tokens = self._estimator.estimate_messages(original)
        selected_tokens = self._context_tokens(selected, summary)
        summary_tokens = self._estimator.estimate_messages([_summary_message(summary)]) if summary else 0
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
            summary_used=summary is not None,
            summary_cache_hit=cache_hit,
            summarized_message_count=summarized_count,
            raw_message_count=len(selected),
            summary_fact_count=len(summary.facts) if summary else 0,
            summary_tokens=summary_tokens,
            summary_fallback_reason=fallback_reason,
        )
        logger.info(
            "context_selection component=%s strategy=%s budget_tokens=%d original_tokens=%d "
            "selected_tokens=%d compression_ratio=%.6f selected_message_count=%d dropped_message_count=%d "
            "summary_used=%s summary_cache_hit=%s summarized_message_count=%d raw_message_count=%d "
            "summary_fact_count=%d summary_tokens=%d summary_fallback_reason=%s",
            metrics.component,
            metrics.strategy,
            metrics.budget_tokens,
            metrics.original_tokens,
            metrics.selected_tokens,
            metrics.compression_ratio,
            metrics.selected_message_count,
            metrics.dropped_message_count,
            metrics.summary_used,
            metrics.summary_cache_hit,
            metrics.summarized_message_count,
            metrics.raw_message_count,
            metrics.summary_fact_count,
            metrics.summary_tokens,
            metrics.summary_fallback_reason or "none",
        )
        return ContextSelection(
            request=UserQuery(
                request_id=request.request_id,
                query=request.query,
                history=selected,
            ),
            metrics=metrics,
            summary=summary,
        )

    def clear_summary_cache(self, request_id: object | None = None) -> None:
        with self._cache_lock:
            if request_id is None:
                self._summary_cache.clear()
                return
            value = str(request_id)
            for key in [key for key, item in self._summary_cache.items() if item[0] == value]:
                del self._summary_cache[key]

    def _summary_compression(self, request: UserQuery, policy: ContextPolicy):
        original_tokens = self._estimator.estimate_messages(request.history)
        if original_tokens <= policy.budget_tokens:
            return list(request.history), None, False, 0, None
        if self._summarizer is None or policy.budget_tokens == 0 or policy.summary_budget_ratio == 0:
            return self._summary_fallback(request, policy, "invalid_summary")

        turns = _group_indexed_turns(request.history)
        recent_count = min(policy.summary_recent_n, len(turns))
        prefix = list(turns[:-recent_count]) if recent_count else list(turns)
        recent = list(turns[-recent_count:]) if recent_count else []
        raw_target = int(policy.budget_tokens * (1 - policy.summary_budget_ratio))
        raw = _flatten_indexed(recent)
        if self._estimator.estimate_messages([message for _, message in raw]) > policy.budget_tokens:
            while recent and self._estimator.estimate_messages(
                [message for _, message in _flatten_indexed(recent)]
            ) > raw_target:
                prefix.append(recent.pop(0))
            raw = _flatten_indexed(recent)
        summarized = _flatten_indexed(prefix)
        if not summarized:
            return self._summary_fallback(request, policy, "empty_summary")

        key = _summary_cache_key(str(request.request_id), request.query, summarized, raw)
        cached = self._cache_get(key)
        cache_hit = cached is not None
        try:
            complete = cached or self._summarizer.summarize(request.query, summarized, raw)
        except ProtectedFactMissingError:
            return self._summary_fallback(
                request, policy, "protected_fact_missing", summarized_count=len(summarized),
            )
        except InvalidHistorySummaryError:
            return self._summary_fallback(
                request, policy, "invalid_summary", summarized_count=len(summarized),
            )
        except SummaryProviderError:
            return self._summary_fallback(
                request, policy, "provider_error", summarized_count=len(summarized),
            )
        if not complete.facts:
            return self._summary_fallback(
                request, policy, "empty_summary", summarized_count=len(summarized),
            )
        if cached is None:
            self._cache_put(key, str(request.request_id), complete)

        raw_messages = [message for _, message in raw]
        packed = self._pack_summary(complete, raw_messages, summarized, raw, policy.budget_tokens)
        if packed is None:
            return self._summary_fallback(
                request, policy, "over_budget", summarized_count=len(summarized), cache_hit=cache_hit,
            )
        summarized_indexes = {index for index, _ in summarized}
        protected = [
            item for item in extract_protected_facts([*summarized, *raw])
            if item.source_message_index in summarized_indexes
        ]
        if any(not _protected_covered(item, packed) for item in protected):
            return self._summary_fallback(
                request, policy, "protected_fact_missing",
                summarized_count=len(summarized), cache_hit=cache_hit,
            )
        return raw_messages, packed, cache_hit, len(summarized), None

    def _pack_summary(
        self,
        complete: HistorySummary,
        raw: list[Message],
        summarized: Sequence[tuple[int, Message]],
        recent: Sequence[tuple[int, Message]],
        budget_tokens: int,
    ) -> HistorySummary | None:
        summarized_indexes = {index for index, _ in summarized}
        protected = [
            item for item in extract_protected_facts([*summarized, *recent])
            if item.source_message_index in summarized_indexes
        ]
        category_order = {"constraint": 0, "entity": 1, "time_range": 2, "confirmed_intent": 3, "planning_fact": 4}
        indexed = list(enumerate(complete.facts))
        indexed.sort(key=lambda item: (
            0 if any(_fact_covers(value, item[1]) for value in protected) else 1,
            category_order[item[1].category],
            item[0],
        ))
        packed: list[SummaryFact] = []
        for _, fact in indexed:
            candidate = HistorySummary(facts=[*packed, fact])
            if self._context_tokens(raw, candidate) <= budget_tokens:
                packed.append(fact)
        return HistorySummary(facts=packed) if packed else None

    def _context_tokens(self, raw: Sequence[Message], summary: HistorySummary | None) -> int:
        messages = [*([_summary_message(summary)] if summary else []), *raw]
        return self._estimator.estimate_messages(messages)

    def _summary_fallback(
        self, request, policy, reason, *, summarized_count: int = 0, cache_hit: bool = False,
    ):
        selected = self._budgeted_selection(request.query, request.history, policy.budget_tokens)
        return selected, None, cache_hit, summarized_count, reason

    def _cache_get(self, key: str) -> HistorySummary | None:
        with self._cache_lock:
            item = self._summary_cache.get(key)
            if item is not None:
                self._summary_cache.move_to_end(key)
                return item[1]
        return None

    def _cache_put(self, key: str, request_id: str, summary: HistorySummary) -> None:
        if self._summary_cache_size <= 0:
            return
        with self._cache_lock:
            self._summary_cache[key] = (request_id, summary)
            self._summary_cache.move_to_end(key)
            while len(self._summary_cache) > self._summary_cache_size:
                self._summary_cache.popitem(last=False)

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


def _group_indexed_turns(history: Sequence[Message]) -> list[list[tuple[int, Message]]]:
    turns: list[list[tuple[int, Message]]] = []
    current: list[tuple[int, Message]] = []
    for index, message in enumerate(history):
        if message.role == "user":
            if current:
                turns.append(current)
            current = [(index, message)]
        else:
            current.append((index, message))
    if current:
        turns.append(current)
    return turns


def _flatten_indexed(turns):
    return [item for turn in turns for item in turn]


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


def _summary_message(summary: HistorySummary) -> Message:
    content = json.dumps(
        {"type": "history_summary", **summary.model_dump(mode="json")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return Message(role="assistant", content=content)


def _summary_cache_key(request_id, query, summarized, recent) -> str:
    payload = {
        "version": SUMMARY_SCHEMA_VERSION,
        "request_id": request_id,
        "query": query,
        "summarized": [(index, message.model_dump(mode="json")) for index, message in summarized],
        "recent": [(index, message.model_dump(mode="json")) for index, message in recent],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fact_covers(protected: ProtectedFact, fact: SummaryFact) -> bool:
    return (
        fact.source_message_index == protected.source_message_index
        and protected.value.lower() in fact.content.lower()
    )


def _protected_covered(protected: ProtectedFact, summary: HistorySummary) -> bool:
    return any(_fact_covers(protected, fact) for fact in summary.facts)
