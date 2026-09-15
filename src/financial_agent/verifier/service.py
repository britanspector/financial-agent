"""Structured evidence-sufficiency verifier independent of concrete providers."""

from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, ValidationError

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.verifier.evidence import InvalidEvidenceInputError, resolve_evidence, validate_execution
from financial_agent.schemas import UserQuery
from financial_agent.verifier.models import DraftAnswer, VerificationResult, VerifierModelOutput
from financial_agent.verifier.prompt import build_verifier_messages, verifier_response_schema
from financial_agent.verifier.providers import VerifierProvider, VerifierProviderResponseError


class OutputCatalog(Protocol):
    def output_model(self, name: str) -> type[BaseModel] | None: ...


class InvalidVerificationInputError(ValueError):
    """Raised before the provider is called when execution evidence is inconsistent."""


class StructuredVerifier:
    def __init__(self, provider: VerifierProvider, catalog: OutputCatalog) -> None:
        self._provider = provider
        self._catalog = catalog

    def verify(
        self,
        request: UserQuery,
        plan: Sequence[Task],
        tool_results: Sequence[TaskExecutionResult],
        draft: DraftAnswer,
    ) -> VerificationResult:
        tasks = list(plan)
        results = list(tool_results)
        try:
            result_by_id = validate_execution(tasks, results)
            resolved_evidence = resolve_evidence(draft.evidence, tasks, results, self._catalog)
        except InvalidEvidenceInputError as exc:
            raise InvalidVerificationInputError(str(exc)) from exc
        ordered_results = [result_by_id[task.task_id] for task in tasks]
        failed_task_ids = [task.task_id for task in tasks if result_by_id[task.task_id].result.status == "error"]
        raw = self._provider.generate(
            build_verifier_messages(
                request,
                tasks,
                ordered_results,
                draft,
                resolved_evidence=resolved_evidence,
                failed_task_ids=failed_task_ids,
            ),
            response_schema=verifier_response_schema(),
        )
        try:
            output = VerifierModelOutput.model_validate(raw)
        except ValidationError as exc:
            raise VerifierProviderResponseError("Invalid verifier response") from exc
        return VerificationResult(**output.model_dump(), failed_task_ids=failed_task_ids)
