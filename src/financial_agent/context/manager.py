"""Deterministic, budget-aware history context management."""

from __future__ import annotations

import logging
import re
import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock

from financial_agent.context.models import (
    ContextComponent,
    ContextMetrics,
    ContextPolicy,
    ContextSelection,
    HistorySummary,
    RetrievedHistoryTurn,
    SummaryFact,
)
from financial_agent.context.retrieval import (
    HistoryRetrievalResult,
    HistoryRetriever,
    HistoryTurn,
    LexicalHistoryRetriever,
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


@dataclass
class _ContextParts:
    selected: list[Message]
    summary: HistorySummary | None = None
    retrieved: list[RetrievedHistoryTurn] = field(default_factory=list)
    cache_hit: bool = False
    summarized_count: int = 0
    summary_fallback_reason: str | None = None
    retrieval_active: bool = False
    retrieval_candidate_count: int = 0
    retrieval_eligible_count: int = 0
    retrieval_fallback_reason: str | None = None
    protected_count: int = 0
    protected_covered_count: int = 0


class ContextManager:
    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        *,
        summarizer: HistorySummarizer | None = None,
        retriever: HistoryRetriever | None = None,
        summary_cache_size: int = 128,
    ) -> None:
        if summary_cache_size < 0:
            raise ValueError("summary_cache_size must be non-negative")
        self._estimator = estimator or HeuristicTokenEstimator()
        self._summarizer = summarizer
        self._retriever = retriever or LexicalHistoryRetriever()
        self._summary_cache_size = summary_cache_size
        self._summary_cache: OrderedDict[str, HistorySummary] = OrderedDict()
        self._cache_lock = Lock()

    def select(
        self,
        request: UserQuery,
        component: ContextComponent,
        policy: ContextPolicy,
    ) -> ContextSelection:
        original = list(request.history)
        if policy.strategy == "full_history":
            parts = _ContextParts(original)
        elif policy.strategy == "last_n":
            turns = _group_turns(original)
            parts = _ContextParts(_flatten(turns[-policy.last_n:]) if policy.last_n else [])
        elif policy.strategy == "budgeted_selection":
            parts = _ContextParts(self._budgeted_selection(request.query, original, policy.budget_tokens))
        elif policy.strategy == "summary_compression":
            parts = self._summary_compression(request, policy)
        else:
            parts = self._summary_retrieval(request, policy)

        original_tokens = self._estimator.estimate_messages(original)
        selected_tokens = self._context_tokens(parts.selected, parts.summary, parts.retrieved)
        summary_tokens = self._estimator.estimate_messages([_summary_message(parts.summary)]) if parts.summary else 0
        retrieved_tokens = self._retrieved_tokens(parts.retrieved)
        ratio = selected_tokens / original_tokens if original_tokens else 1.0
        selected_message_count = len(parts.selected) + sum(len(turn.messages) for turn in parts.retrieved)
        metrics = ContextMetrics(
            component=component,
            strategy=policy.strategy,
            budget_tokens=policy.budget_tokens,
            original_tokens=original_tokens,
            selected_tokens=selected_tokens,
            compression_ratio=ratio,
            selected_message_count=selected_message_count,
            dropped_message_count=len(original) - selected_message_count,
            summary_used=parts.summary is not None,
            summary_cache_hit=parts.cache_hit,
            summarized_message_count=parts.summarized_count,
            raw_message_count=selected_message_count,
            summary_fact_count=len(parts.summary.facts) if parts.summary else 0,
            summary_tokens=summary_tokens,
            summary_fallback_reason=parts.summary_fallback_reason,
            retrieval_active=parts.retrieval_active,
            retrieval_candidate_turn_count=parts.retrieval_candidate_count,
            retrieval_eligible_turn_count=parts.retrieval_eligible_count,
            retrieved_turn_count=len(parts.retrieved),
            retrieved_message_count=sum(len(turn.messages) for turn in parts.retrieved),
            retrieved_tokens=retrieved_tokens,
            recent_message_count=len(parts.selected),
            retrieval_fallback_reason=parts.retrieval_fallback_reason,
            protected_fact_count=parts.protected_count,
            protected_fact_covered_count=parts.protected_covered_count,
        )
        logger.info(
            "context_selection component=%s strategy=%s budget_tokens=%d original_tokens=%d "
            "selected_tokens=%d compression_ratio=%.6f selected_message_count=%d dropped_message_count=%d "
            "summary_used=%s summary_cache_hit=%s summarized_message_count=%d raw_message_count=%d "
            "summary_fact_count=%d summary_tokens=%d summary_fallback_reason=%s "
            "retrieval_active=%s retrieval_candidate_turn_count=%d retrieval_eligible_turn_count=%d "
            "retrieved_turn_count=%d retrieved_message_count=%d retrieved_tokens=%d recent_message_count=%d "
            "retrieval_fallback_reason=%s protected_fact_count=%d protected_fact_covered_count=%d",
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
            metrics.retrieval_active,
            metrics.retrieval_candidate_turn_count,
            metrics.retrieval_eligible_turn_count,
            metrics.retrieved_turn_count,
            metrics.retrieved_message_count,
            metrics.retrieved_tokens,
            metrics.recent_message_count,
            metrics.retrieval_fallback_reason or "none",
            metrics.protected_fact_count,
            metrics.protected_fact_covered_count,
        )
        return ContextSelection(
            request=UserQuery(
                request_id=request.request_id,
                query=request.query,
                history=parts.selected,
            ),
            metrics=metrics,
            summary=parts.summary,
            retrieved_history=parts.retrieved,
        )

    def clear_summary_cache(self, request_id: object | None = None) -> None:
        del request_id
        with self._cache_lock:
            self._summary_cache.clear()

    def _summary_compression(self, request: UserQuery, policy: ContextPolicy):
        original_tokens = self._estimator.estimate_messages(request.history)
        if original_tokens <= policy.budget_tokens:
            return _ContextParts(list(request.history))
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

        key = _summary_cache_key(self._summarizer.cache_discriminator, summarized)
        cached = self._cache_get(key)
        cache_hit = cached is not None
        try:
            complete = cached or self._summarizer.summarize(summarized)
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
            self._cache_put(key, complete)

        raw_messages = [message for _, message in raw]
        active_complete = _filter_stale_summary(complete, summarized, raw)
        packed = self._pack_summary(active_complete, raw_messages, summarized, raw, policy.budget_tokens)
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
        return _ContextParts(
            raw_messages, packed, cache_hit=cache_hit, summarized_count=len(summarized),
        )

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

    def _context_tokens(
        self,
        raw: Sequence[Message],
        summary: HistorySummary | None,
        retrieved: Sequence[RetrievedHistoryTurn] = (),
    ) -> int:
        messages = [
            *([_summary_message(summary)] if summary else []),
            *([_retrieval_message(retrieved)] if retrieved else []),
            *raw,
        ]
        return self._estimator.estimate_messages(messages)

    def _retrieved_tokens(self, retrieved: Sequence[RetrievedHistoryTurn]) -> int:
        return self._estimator.estimate_messages([_retrieval_message(retrieved)]) if retrieved else 0

    def _summary_fallback(
        self, request, policy, reason, *, summarized_count: int = 0, cache_hit: bool = False,
    ):
        selected = self._budgeted_selection(request.query, request.history, policy.budget_tokens)
        return _ContextParts(
            selected,
            cache_hit=cache_hit,
            summarized_count=summarized_count,
            summary_fallback_reason=reason,
        )

    def _cache_get(self, key: str) -> HistorySummary | None:
        with self._cache_lock:
            item = self._summary_cache.get(key)
            if item is not None:
                self._summary_cache.move_to_end(key)
                return item
        return None

    def _cache_put(self, key: str, summary: HistorySummary) -> None:
        if self._summary_cache_size <= 0:
            return
        with self._cache_lock:
            self._summary_cache[key] = summary
            self._summary_cache.move_to_end(key)
            while len(self._summary_cache) > self._summary_cache_size:
                self._summary_cache.popitem(last=False)

    def _summary_retrieval(self, request: UserQuery, policy: ContextPolicy) -> _ContextParts:
        if self._estimator.estimate_messages(request.history) <= policy.budget_tokens:
            return _ContextParts(list(request.history))
        if policy.budget_tokens == 0:
            parts = self._summary_fallback(request, policy, "over_budget")
            parts.retrieval_fallback_reason = "protected_fact_over_budget"
            return parts

        turns = _history_turns(request.history)
        recent_count = min(policy.summary_recent_n, len(turns))
        prefix_turns = turns[:-recent_count] if recent_count else turns
        recent_turns = turns[-recent_count:] if recent_count else []
        summarized = [
            (index, message)
            for turn in prefix_turns
            for index, message in zip(turn.message_indexes, turn.messages)
        ]
        recent_indexed = [
            (index, message)
            for turn in recent_turns
            for index, message in zip(turn.message_indexes, turn.messages)
        ]

        complete_summary = None
        cache_hit = False
        summary_reason = None
        if self._summarizer is not None and summarized:
            key = _summary_cache_key(self._summarizer.cache_discriminator, summarized)
            complete_summary = self._cache_get(key)
            cache_hit = complete_summary is not None
            if complete_summary is None:
                try:
                    complete_summary = self._summarizer.summarize(summarized)
                    if complete_summary.facts:
                        self._cache_put(key, complete_summary)
                    else:
                        complete_summary = None
                        summary_reason = "empty_summary"
                except ProtectedFactMissingError:
                    summary_reason = "protected_fact_missing"
                except InvalidHistorySummaryError:
                    summary_reason = "invalid_summary"
                except SummaryProviderError:
                    summary_reason = "provider_error"
        elif summarized:
            summary_reason = "invalid_summary"
        active_summary = (
            _filter_stale_summary(complete_summary, summarized, recent_indexed)
            if complete_summary is not None else None
        )

        retrieval_result = HistoryRetrievalResult((), len(prefix_turns), 0)
        retrieval_error = False
        try:
            retrieval_result = self._retriever.retrieve(
                request.query,
                prefix_turns,
                stable_summary=active_summary,
                full_history=request.history,
                top_k=policy.retrieval_top_k,
                min_score=policy.retrieval_min_score,
            )
        except Exception:  # Custom retrievers are isolated from the Agent Loop.
            retrieval_error = True

        if active_summary is None and not retrieval_result.hits:
            parts = self._summary_fallback(
                request, policy, summary_reason or "invalid_summary", summarized_count=len(summarized),
                cache_hit=cache_hit,
            )
            parts.retrieval_candidate_count = retrieval_result.candidate_count
            parts.retrieval_eligible_count = retrieval_result.eligible_count
            parts.retrieval_fallback_reason = "both_failed"
            return parts

        packed = self._pack_retrieval_context(
            request.history,
            recent_turns,
            prefix_turns,
            list(retrieval_result.hits),
            active_summary,
            policy,
        )
        if packed is None:
            parts = self._summary_fallback(
                request, policy, "over_budget", summarized_count=len(summarized), cache_hit=cache_hit,
            )
            parts.retrieval_candidate_count = retrieval_result.candidate_count
            parts.retrieval_eligible_count = retrieval_result.eligible_count
            parts.retrieval_fallback_reason = "protected_fact_over_budget"
            return parts
        recent_messages, retrieved, summary, protected_count, covered_count = packed
        return _ContextParts(
            selected=recent_messages,
            summary=summary,
            retrieved=retrieved,
            cache_hit=cache_hit,
            summarized_count=len(summarized),
            summary_fallback_reason=summary_reason,
            retrieval_active=any(
                item.score > 0 and item.score >= policy.retrieval_min_score for item in retrieved
            ),
            retrieval_candidate_count=retrieval_result.candidate_count,
            retrieval_eligible_count=retrieval_result.eligible_count,
            retrieval_fallback_reason=(
                "retriever_error" if retrieval_error
                else "summary_error" if active_summary is None
                else None
            ),
            protected_count=protected_count,
            protected_covered_count=covered_count,
        )

    def _pack_retrieval_context(
        self,
        history: Sequence[Message],
        recent_turns: Sequence[HistoryTurn],
        prefix_turns: Sequence[HistoryTurn],
        hits: list[RetrievedHistoryTurn],
        summary: HistorySummary | None,
        policy: ContextPolicy,
    ) -> tuple[list[Message], list[RetrievedHistoryTurn], HistorySummary | None, int, int] | None:
        budget = policy.budget_tokens
        recent_cap = int(budget * policy.retrieval_recent_reservation_ratio)
        retrieval_cap = int(budget * policy.retrieval_history_reservation_ratio) if hits else 0
        chosen_recent: list[HistoryTurn] = []
        chosen_retrieved: list[RetrievedHistoryTurn] = []
        chosen_facts: list[SummaryFact] = []

        for turn in reversed(recent_turns):
            candidate = sorted([*chosen_recent, turn], key=lambda item: item.turn_index)
            if self._estimator.estimate_messages(_turn_messages(candidate)) <= recent_cap:
                chosen_recent.append(turn)
        for hit in hits:
            candidate = [*chosen_retrieved, hit]
            if self._retrieved_tokens(candidate) <= retrieval_cap:
                chosen_retrieved.append(hit)

        protected = extract_protected_facts(list(enumerate(history)))
        turn_by_message = {
            index: turn for turn in [*prefix_turns, *recent_turns] for index in turn.message_indexes
        }
        summary_facts = list(summary.facts) if summary else []
        for protected_fact in protected:
            if _protected_in_raw(protected_fact, chosen_recent, chosen_retrieved):
                continue
            fact = next((item for item in summary_facts if _fact_covers(protected_fact, item)), None)
            if fact is not None:
                if fact not in chosen_facts:
                    chosen_facts.append(fact)
                if not self._fits_retrieval_context(
                    chosen_recent, chosen_retrieved, chosen_facts, budget,
                ):
                    return None
                continue
            turn = turn_by_message.get(protected_fact.source_message_index)
            if turn is None:
                return None
            if turn in recent_turns:
                if turn not in chosen_recent:
                    chosen_recent.append(turn)
            else:
                if not any(item.turn_index == turn.turn_index for item in chosen_retrieved):
                    chosen_retrieved.append(_protected_retrieved_turn(turn, chosen_retrieved))
            if not self._fits_retrieval_context(chosen_recent, chosen_retrieved, chosen_facts, budget):
                return None

        if hits and not any(item.score > 0 for item in chosen_retrieved):
            for hit in hits:
                candidate_facts = [
                    fact for fact in chosen_facts if fact.source_message_index not in hit.message_indexes
                ]
                candidate = [*chosen_retrieved, hit]
                if self._fits_retrieval_context(chosen_recent, candidate, candidate_facts, budget):
                    chosen_retrieved.append(hit)
                    chosen_facts = candidate_facts
                    break

        for turn in reversed(recent_turns):
            if turn in chosen_recent:
                continue
            candidate = [*chosen_recent, turn]
            candidate_facts = [
                fact for fact in chosen_facts if fact.source_message_index not in turn.message_indexes
            ]
            if self._fits_retrieval_context(candidate, chosen_retrieved, candidate_facts, budget):
                chosen_recent.append(turn)
                chosen_facts = candidate_facts
        for hit in hits:
            if any(item.turn_index == hit.turn_index for item in chosen_retrieved):
                continue
            candidate = [*chosen_retrieved, hit]
            candidate_facts = [
                fact for fact in chosen_facts if fact.source_message_index not in hit.message_indexes
            ]
            if self._fits_retrieval_context(chosen_recent, candidate, candidate_facts, budget):
                chosen_retrieved.append(hit)
                chosen_facts = candidate_facts
        category_order = {"constraint": 0, "entity": 1, "time_range": 2, "confirmed_intent": 3, "planning_fact": 4}
        for fact in sorted(summary_facts, key=lambda item: (category_order[item.category], item.source_message_index)):
            if fact in chosen_facts or _fact_source_in_raw(fact, chosen_recent, chosen_retrieved):
                continue
            candidate = [*chosen_facts, fact]
            if self._fits_retrieval_context(chosen_recent, chosen_retrieved, candidate, budget):
                chosen_facts.append(fact)

        chosen_recent.sort(key=lambda item: item.turn_index)
        chosen_retrieved.sort(key=lambda item: (-item.score, item.rank, -item.turn_index))
        chosen_retrieved = [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(chosen_retrieved, 1)
        ]
        packed_summary = HistorySummary(facts=chosen_facts) if chosen_facts else None
        covered = sum(
            _protected_in_raw(item, chosen_recent, chosen_retrieved)
            or (packed_summary is not None and _protected_covered(item, packed_summary))
            for item in protected
        )
        if covered != len(protected):
            return None
        return _turn_messages(chosen_recent), chosen_retrieved, packed_summary, len(protected), covered

    def _fits_retrieval_context(self, recent, retrieved, facts, budget) -> bool:
        packed_summary = HistorySummary(facts=facts) if facts else None
        return self._context_tokens(_turn_messages(recent), packed_summary, retrieved) <= budget

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


def _retrieval_message(retrieved: Sequence[RetrievedHistoryTurn]) -> Message:
    payload = [{
        "rank": item.rank,
        "message_indexes": item.message_indexes,
        "messages": [message.model_dump(mode="json") for message in item.messages],
    } for item in retrieved]
    return Message(
        role="assistant",
        content=json.dumps(
            {"type": "retrieved_history", "turns": payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _summary_cache_key(cache_discriminator: str, summarized) -> str:
    payload = {
        "version": SUMMARY_SCHEMA_VERSION,
        "summarizer": cache_discriminator,
        "summarized": [(index, message.model_dump(mode="json")) for index, message in summarized],
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


def _history_turns(history: Sequence[Message]) -> list[HistoryTurn]:
    return [
        HistoryTurn(
            turn_index=turn_index,
            message_indexes=tuple(index for index, _ in turn),
            messages=tuple(message for _, message in turn),
        )
        for turn_index, turn in enumerate(_group_indexed_turns(history))
    ]


def _turn_messages(turns: Sequence[HistoryTurn]) -> list[Message]:
    return [message for turn in turns for message in turn.messages]


def _protected_in_raw(
    protected: ProtectedFact,
    recent: Sequence[HistoryTurn],
    retrieved: Sequence[RetrievedHistoryTurn],
) -> bool:
    indexes = {
        index for turn in recent for index in turn.message_indexes
    } | {
        index for turn in retrieved for index in turn.message_indexes
    }
    return protected.source_message_index in indexes


def _fact_source_in_raw(
    fact: SummaryFact,
    recent: Sequence[HistoryTurn],
    retrieved: Sequence[RetrievedHistoryTurn],
) -> bool:
    return _protected_in_raw(
        ProtectedFact(fact.category, fact.content, fact.source_message_index), recent, retrieved,
    )


def _protected_retrieved_turn(
    turn: HistoryTurn,
    chosen: Sequence[RetrievedHistoryTurn],
) -> RetrievedHistoryTurn:
    return RetrievedHistoryTurn(
        rank=max([item.rank for item in chosen], default=0) + 1,
        score=0,
        turn_index=turn.turn_index,
        message_indexes=list(turn.message_indexes),
        messages=list(turn.messages),
    )


def _filter_stale_summary(
    summary: HistorySummary,
    summarized: Sequence[tuple[int, Message]],
    later: Sequence[tuple[int, Message]],
) -> HistorySummary:
    prefix_protected = extract_protected_facts(summarized)
    effective = extract_protected_facts([*summarized, *later])
    effective_keys = {
        (item.category, item.value.casefold(), item.source_message_index) for item in effective
    }
    stale = [
        item for item in prefix_protected
        if (item.category, item.value.casefold(), item.source_message_index) not in effective_keys
    ]
    return HistorySummary(facts=[
        fact for fact in summary.facts
        if not any(_fact_covers(item, fact) for item in stale)
    ])
