"""Deterministic lexical retrieval over complete conversation turns."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from financial_agent.context.models import HistorySummary, RetrievedHistoryTurn
from financial_agent.knowledge.tokenization import ChineseTokenizer
from financial_agent.schemas import Message


_USER_ID = re.compile(r"\bsyn-user-\d{4}\b|\buser[_-]?id\s*[:=：]?\s*[a-z0-9._-]+\b", re.IGNORECASE)
_SYMBOL = re.compile(r"(?<!\d)\d{6}(?:\.(?:sh|sz))?(?!\d)", re.IGNORECASE)
_DATE = re.compile(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)|(?<!\d)\d{4}年\d{1,2}月\d{1,2}日")
_CORRECTION = re.compile(r"更正|改为|不是.+是|以.+为准|correct(?:ion|ed)?|instead", re.IGNORECASE)
_REFERENCE = re.compile(r"它|他|该公司|之前|前面|继续|按原来|按之前|same|previous|continue", re.IGNORECASE)
_CONSTRAINT = re.compile(r"不要|不得|不能|必须|务必|仅|只(?:要|分析|使用|需要)?", re.IGNORECASE)
_CONSTRAINT_REVERSAL = re.compile(r"取消(?:之前|原来)?.{0,12}(?:限制|要求)|不再(?:要求|限制)|更正.{0,20}可以", re.IGNORECASE)


@dataclass(frozen=True)
class HistoryTurn:
    turn_index: int
    message_indexes: tuple[int, ...]
    messages: tuple[Message, ...]

    @property
    def text(self) -> str:
        return "\n".join(message.content for message in self.messages)


@dataclass(frozen=True)
class HistoryRetrievalResult:
    hits: tuple[RetrievedHistoryTurn, ...]
    candidate_count: int
    eligible_count: int


class HistoryRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        candidates: Sequence[HistoryTurn],
        *,
        stable_summary: HistorySummary | None,
        full_history: Sequence[Message],
        top_k: int,
        min_score: float,
    ) -> HistoryRetrievalResult: ...


class LexicalHistoryRetriever:
    """Request-local BM25 with deterministic entity boosts and ordering."""

    def __init__(self, tokenizer: ChineseTokenizer | None = None) -> None:
        self._tokenizer = tokenizer or ChineseTokenizer()

    def retrieve(
        self,
        query: str,
        candidates: Sequence[HistoryTurn],
        *,
        stable_summary: HistorySummary | None,
        full_history: Sequence[Message],
        top_k: int,
        min_score: float,
    ) -> HistoryRetrievalResult:
        if not candidates or top_k <= 0:
            return HistoryRetrievalResult((), len(candidates), 0)
        filtered = [turn for turn in candidates if not _contains_superseded_value(turn, full_history)]
        expanded_query = self._expand_query(query, stable_summary)
        query_terms = self._tokenizer.tokenize(expanded_query)
        query_term_set = set(query_terms)
        query_entities = _entities(query)
        index = _TurnBM25(filtered, self._tokenizer)
        ranked: list[tuple[float, int, HistoryTurn]] = []
        eligible_count = 0
        for position, turn in enumerate(filtered):
            turn_terms = set(self._tokenizer.tokenize(turn.text))
            shared_entities = len(query_entities & _entities(turn.text))
            if not (query_term_set & turn_terms) and shared_entities == 0:
                continue
            eligible_count += 1
            raw_bm25 = index.score(query_terms, position)
            score = raw_bm25 / (1.0 + raw_bm25) + 0.25 * min(shared_entities, 2)
            if score >= min_score:
                ranked.append((score, turn.turn_index, turn))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        hits = tuple(
            RetrievedHistoryTurn(
                rank=rank,
                score=score,
                turn_index=turn.turn_index,
                message_indexes=list(turn.message_indexes),
                messages=list(turn.messages),
            )
            for rank, (score, _, turn) in enumerate(ranked[:top_k], 1)
        )
        return HistoryRetrievalResult(hits, len(candidates), eligible_count)

    def _expand_query(self, query: str, summary: HistorySummary | None) -> str:
        if summary is None or (_entities(query) and not _REFERENCE.search(query)):
            return query
        additions = [
            fact.content for fact in summary.facts
            if fact.category in {"entity", "time_range", "confirmed_intent"}
        ][:8]
        return "\n".join([query, *additions])


def retrieved_history_payload(turns: Sequence[RetrievedHistoryTurn]) -> list[dict[str, object]]:
    return [{
        "rank": turn.rank,
        "turn_index": turn.turn_index,
        "message_indexes": turn.message_indexes,
        "messages": [message.model_dump(mode="json") for message in turn.messages],
    } for turn in turns]


class _TurnBM25:
    def __init__(self, turns: Sequence[HistoryTurn], tokenizer: ChineseTokenizer) -> None:
        self._frequencies = [Counter(tokenizer.tokenize(turn.text)) for turn in turns]
        self._lengths = [sum(item.values()) for item in self._frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())
        count = len(turns)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query_terms: list[str], position: int) -> float:
        frequencies = self._frequencies[position]
        length = self._lengths[position]
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * length / self._average_length
            ) if self._average_length else frequency
            score += self._idf.get(term, 0.0) * frequency * 2.5 / denominator * query_frequency
        return score


def _entities(text: str) -> set[str]:
    values: set[str] = set()
    for category, pattern in (("user_id", _USER_ID), ("symbol", _SYMBOL), ("date", _DATE)):
        values.update(f"{category}:{match.group(0).casefold()}" for match in pattern.finditer(text))
    return values


def _contains_superseded_value(turn: HistoryTurn, history: Sequence[Message]) -> bool:
    later_text = "\n".join(message.content for message in history[turn.message_indexes[-1] + 1:] if message.role == "user")
    if not later_text:
        return False
    for pattern in (_USER_ID, _SYMBOL, _DATE):
        values = {match.group(0).casefold() for match in pattern.finditer(turn.text)}
        later_values = {match.group(0).casefold() for match in pattern.finditer(later_text)}
        if values and later_values and values != later_values and _CORRECTION.search(later_text):
            return True
    if _CONSTRAINT.search(turn.text) and _CONSTRAINT_REVERSAL.search(later_text):
        turn_terms = set(ChineseTokenizer().tokenize(turn.text))
        later_terms = set(ChineseTokenizer().tokenize(later_text))
        if len(turn_terms & later_terms) >= 2:
            return True
    return False
