"""Grounded structured history summarization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from financial_agent.context.models import HistorySummary, HistorySummaryUpdate, SummaryFact
from financial_agent.context.summary_prompt import (
    build_incremental_summary_messages,
    build_summary_messages,
    summary_response_schema,
    summary_update_response_schema,
)
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
_CONSTRAINT_REVERSAL = re.compile(
    r"取消(?:之前|原来)?.{0,12}(?:限制|要求)|不再(?:要求|限制)|更正.{0,20}(?:可以|允许)",
    re.IGNORECASE,
)
_TERM = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.IGNORECASE)


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

    @property
    def cache_discriminator(self) -> str:
        from financial_agent.context.summary_prompt import SUMMARY_SCHEMA_VERSION
        return f"{SUMMARY_SCHEMA_VERSION}:{self._max_facts}"

    def summarize(
        self,
        summarized_messages: Sequence[tuple[int, Message]],
    ) -> HistorySummary:
        raw = self._provider.generate(
            build_summary_messages(summarized_messages),
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
        missing = [
            fact for fact in extract_protected_facts(summarized_messages)
            if not _covered(fact, summary)
        ]
        if missing:
            raise ProtectedFactMissingError("Summary omitted protected history facts")
        return summary

    def update(
        self,
        existing_summary: HistorySummary,
        existing_messages: Sequence[tuple[int, Message]],
        newly_aged_out_messages: Sequence[tuple[int, Message]],
    ) -> HistorySummary:
        if not newly_aged_out_messages:
            return existing_summary
        raw = self._provider.generate(
            self.incremental_messages(existing_summary, newly_aged_out_messages),
            response_schema=summary_update_response_schema(self._max_facts),
        )
        try:
            update = HistorySummaryUpdate.model_validate(raw)
        except ValidationError as exc:
            raise InvalidHistorySummaryError("Invalid incremental summary response") from exc
        replacement_indexes = [item.existing_fact_index for item in update.replacements]
        if len(replacement_indexes) != len(set(replacement_indexes)):
            raise InvalidHistorySummaryError("Duplicate incremental replacement index")
        if any(index >= len(existing_summary.facts) for index in replacement_indexes):
            raise InvalidHistorySummaryError("Incremental replacement index is out of range")

        new_by_index = {index: message for index, message in newly_aged_out_messages}
        new_facts = [*update.additions, *(item.fact for item in update.replacements)]
        for fact in new_facts:
            source = new_by_index.get(fact.source_message_index)
            if source is None or fact.content not in source.content:
                raise InvalidHistorySummaryError("Incremental fact is not grounded in newly aged-out history")

        facts = list(existing_summary.facts)
        for replacement in update.replacements:
            facts[replacement.existing_fact_index] = replacement.fact
        facts.extend(update.additions)
        target_messages = [*existing_messages, *newly_aged_out_messages]
        facts = _remove_stale_protected_facts(facts, existing_messages, target_messages)
        facts = _deduplicate_facts(facts)
        if len(facts) > self._max_facts:
            raise InvalidHistorySummaryError("Incremental summary exceeds max facts")
        summary = HistorySummary(facts=facts)
        self._validate_complete_summary(summary, target_messages)
        return summary

    def rebuild_messages(self, summarized_messages: Sequence[tuple[int, Message]]):
        return build_summary_messages(summarized_messages)

    def incremental_messages(
        self,
        existing_summary: HistorySummary,
        newly_aged_out_messages: Sequence[tuple[int, Message]],
    ):
        return build_incremental_summary_messages(
            existing_summary.model_dump(mode="json"), newly_aged_out_messages,
        )

    def _validate_complete_summary(
        self,
        summary: HistorySummary,
        messages: Sequence[tuple[int, Message]],
    ) -> None:
        by_index = {index: message for index, message in messages}
        for fact in summary.facts:
            source = by_index.get(fact.source_message_index)
            if source is None or fact.content not in source.content:
                raise InvalidHistorySummaryError("Summary fact is not grounded in its source message")
        missing = [fact for fact in extract_protected_facts(messages) if not _covered(fact, summary)]
        if missing:
            raise ProtectedFactMissingError("Summary omitted protected history facts")


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
            if clause and _CONSTRAINT_REVERSAL.search(clause):
                correction_terms = _terms(clause)
                by_category["constraint"] = [
                    fact for fact in by_category["constraint"]
                    if not (_terms(fact.value) & correction_terms)
                ]
                by_category["constraint"].append(ProtectedFact("constraint", clause, index))
            elif clause and _CONSTRAINT.search(clause):
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


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _TERM.finditer(text.casefold()):
        value = match.group(0)
        if "\u4e00" <= value[0] <= "\u9fff" and len(value) > 1:
            terms.update(value[index:index + 2] for index in range(len(value) - 1))
        else:
            terms.add(value)
    return terms


def _covered(protected: ProtectedFact, summary: HistorySummary) -> bool:
    return any(
        fact.source_message_index == protected.source_message_index
        and protected.value.lower() in fact.content.lower()
        for fact in summary.facts
    )


def _remove_stale_protected_facts(
    facts: list[SummaryFact],
    previous_messages: Sequence[tuple[int, Message]],
    target_messages: Sequence[tuple[int, Message]],
) -> list[SummaryFact]:
    effective = {
        (fact.category, fact.value.casefold(), fact.source_message_index)
        for fact in extract_protected_facts(target_messages)
    }
    stale = [
        fact for fact in extract_protected_facts(previous_messages)
        if (fact.category, fact.value.casefold(), fact.source_message_index) not in effective
    ]
    return [
        fact for fact in facts
        if not any(
            fact.source_message_index == protected.source_message_index
            and protected.value.casefold() in fact.content.casefold()
            for protected in stale
        )
    ]


def _deduplicate_facts(facts: Sequence[SummaryFact]) -> list[SummaryFact]:
    result: list[SummaryFact] = []
    seen: set[tuple[str, str, int]] = set()
    for fact in facts:
        key = (fact.category, fact.content.casefold(), fact.source_message_index)
        if key not in seen:
            seen.add(key)
            result.append(fact)
    return result
