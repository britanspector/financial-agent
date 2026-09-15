"""Grounded structured history summarization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from financial_agent.context.models import HistorySummary
from financial_agent.context.summary_prompt import build_summary_messages, summary_response_schema
from financial_agent.context.summary_providers import SummaryProvider, SummaryProviderResponseError
from financial_agent.schemas import Message


_USER_ID = re.compile(r"\bsyn-user-\d{4}\b|\buser[_-]?id\s*[:=：]?\s*[a-z0-9._-]+\b", re.IGNORECASE)
_SYMBOL = re.compile(r"(?<!\d)\d{6}(?:\.(?:sh|sz))?(?!\d)", re.IGNORECASE)
_DATE = re.compile(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)|(?<!\d)\d{4}年\d{1,2}月\d{1,2}日")
_CORRECTION = re.compile(r"更正|改为|不是.+是|以.+为准|correct(?:ion|ed)?|instead", re.IGNORECASE)
_CONSTRAINT = re.compile(
    r"必须|务必|仅|只(?:要|分析|使用|需要)?|不要|不得|不能|优先|先.+再|截至|"
    r"\bmust\b|\bonly\b|\bdo\s+not\b|\bdon't\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProtectedFact:
    category: str
    value: str
    source_message_index: int


class InvalidHistorySummaryError(SummaryProviderResponseError):
    pass


class ProtectedFactMissingError(SummaryProviderResponseError):
    pass


class HistorySummarizer:
    def __init__(self, provider: SummaryProvider, *, max_facts: int = 24) -> None:
        self._provider = provider
        self._max_facts = max_facts

    def summarize(
        self,
        query: str,
        summarized_messages: Sequence[tuple[int, Message]],
        recent_messages: Sequence[tuple[int, Message]],
    ) -> HistorySummary:
        raw = self._provider.generate(
            build_summary_messages(query, summarized_messages, recent_messages),
            response_schema=summary_response_schema(self._max_facts),
        )
        try:
            summary = HistorySummary.model_validate(raw)
        except ValidationError as exc:
            raise InvalidHistorySummaryError("Invalid summary response") from exc
        by_index = {index: message for index, message in summarized_messages}
        for fact in summary.facts:
            source = by_index.get(fact.source_message_index)
            if source is None or fact.content not in source.content:
                raise InvalidHistorySummaryError("Summary fact is not grounded in its source message")
        summarized_indexes = {index for index, _ in summarized_messages}
        protected = extract_protected_facts([*summarized_messages, *recent_messages])
        missing = [
            fact for fact in protected
            if fact.source_message_index in summarized_indexes and not _covered(fact, summary)
        ]
        if missing:
            raise ProtectedFactMissingError("Summary omitted protected history facts")
        return summary


def extract_protected_facts(messages: Sequence[tuple[int, Message]]) -> list[ProtectedFact]:
    by_category: dict[str, list[ProtectedFact]] = {"user_id": [], "symbol": [], "date": [], "constraint": []}
    for index, message in messages:
        if message.role != "user":
            continue
        found = {
            "user_id": [match.group(0) for match in _USER_ID.finditer(message.content)],
            "symbol": [match.group(0) for match in _SYMBOL.finditer(message.content)],
            "date": [match.group(0) for match in _DATE.finditer(message.content)],
        }
        if _CORRECTION.search(message.content):
            for category, values in found.items():
                if values:
                    by_category[category] = []
        for category, values in found.items():
            by_category[category].extend(ProtectedFact(category, value, index) for value in values)
        for clause in re.split(r"[。！？；;\n]", message.content):
            clause = clause.strip()
            if clause and _CONSTRAINT.search(clause):
                by_category["constraint"].append(ProtectedFact("constraint", clause, index))
    result: list[ProtectedFact] = []
    seen: set[tuple[str, str, int]] = set()
    for facts in by_category.values():
        for fact in facts:
            key = (fact.category, fact.value.lower(), fact.source_message_index)
            if key not in seen:
                seen.add(key)
                result.append(fact)
    return result


def _covered(protected: ProtectedFact, summary: HistorySummary) -> bool:
    return any(
        fact.source_message_index == protected.source_message_index
        and protected.value.lower() in fact.content.lower()
        for fact in summary.facts
    )
