"""Structured evidence-sufficiency verifier independent of concrete providers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ValidationError

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.agent.result_path import ResultPathError, resolve_result_path, result_path_type
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
        result_by_id = self._validate_execution(tasks, results)
        ordered_results = [result_by_id[task.task_id] for task in tasks]
        failed_task_ids = [task.task_id for task in tasks if result_by_id[task.task_id].result.status == "error"]
        resolved_evidence = [
            self._resolve_evidence(reference.task_id, reference.source_path, tasks, result_by_id)
            for reference in draft.evidence
        ]
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

    @staticmethod
    def _validate_execution(
        plan: list[Task],
        results: list[TaskExecutionResult],
    ) -> dict[str, TaskExecutionResult]:
        if not plan:
            raise InvalidVerificationInputError("Verifier requires a non-empty execute plan")
        task_counts = Counter(task.task_id for task in plan)
        result_counts = Counter(item.task_id for item in results)
        if any(count != 1 for count in task_counts.values()):
            raise InvalidVerificationInputError("Plan task IDs must be unique")
        if any(count != 1 for count in result_counts.values()):
            raise InvalidVerificationInputError("Tool result task IDs must be unique")
        if set(task_counts) != set(result_counts):
            raise InvalidVerificationInputError("Plan and Tool result task IDs must match exactly")
        result_by_id = {item.task_id: item for item in results}
        for task in plan:
            if result_by_id[task.task_id].tool_name != task.tool_name:
                raise InvalidVerificationInputError("Plan and Tool result names must match")
        return result_by_id

    def _resolve_evidence(
        self,
        task_id: str,
        source_path: list[str | int],
        plan: list[Task],
        results: dict[str, TaskExecutionResult],
    ) -> dict[str, Any]:
        task_by_id = {task.task_id: task for task in plan}
        task = task_by_id.get(task_id)
        if task is None:
            raise InvalidVerificationInputError(f"Evidence task does not exist: {task_id}")
        item = results[task_id]
        if item.result.status == "error":
            raise InvalidVerificationInputError(f"Evidence task failed: {task_id}")
        if result_path_type(self._catalog.output_model(task.tool_name), source_path) is None:
            raise InvalidVerificationInputError(f"Evidence path is not a public result field: {task_id}")
        try:
            value = resolve_result_path(item.result.data, source_path)
        except ResultPathError as exc:
            raise InvalidVerificationInputError(f"Evidence value is unavailable: {task_id}") from exc
        return {"task_id": task_id, "source_path": source_path, "value": value}
