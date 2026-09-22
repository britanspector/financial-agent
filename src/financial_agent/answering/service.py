"""Evidence-grounded structured draft writer."""

import json
import re
from copy import deepcopy

from pydantic import BaseModel, ValidationError

from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.answering.prompt import answer_response_schema, build_answer_messages
from financial_agent.answering.providers import AnswerProvider, AnswerProviderResponseError
from financial_agent.context import ContextManager, ContextPolicy
from financial_agent.schemas import UserQuery
from financial_agent.verifier.evidence import InvalidEvidenceInputError, OutputCatalog, resolve_evidence
from financial_agent.verifier.models import DraftAnswer, VerificationResult


class AnswerWriter:
    def __init__(
        self,
        provider: AnswerProvider,
        catalog: OutputCatalog,
        *,
        context_manager: ContextManager | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._catalog = catalog
        self._context_manager = context_manager or ContextManager()
        self._context_policy = context_policy or ContextPolicy()

    def write(self, request: UserQuery, plan: list[Task], results: list[TaskExecutionResult]) -> DraftAnswer:
        return self._generate(request, plan, results)

    def rewrite(
        self, request: UserQuery, plan: list[Task], results: list[TaskExecutionResult],
        draft: DraftAnswer, feedback: VerificationResult,
    ) -> DraftAnswer:
        return self._generate(request, plan, results, previous_draft=draft, feedback=feedback)

    def _generate(self, request, plan, results, *, previous_draft=None, feedback=None) -> DraftAnswer:
        selection = self._context_manager.select(
            request, "writer", self._context_policy,
        )
        contextual_request = selection.request
        messages = build_answer_messages(
            contextual_request,
            plan,
            results,
            previous_draft=previous_draft,
            feedback=feedback,
            history_summary=selection.summary,
            retrieved_history=selection.retrieved_history,
        )
        raw = _generate_traced(self._provider, "writer", messages, answer_response_schema())
        try:
            draft = DraftAnswer.model_validate(_normalize_evidence_paths(raw, results))
            resolve_evidence(draft.evidence, plan, results, self._catalog)
        except (ValidationError, InvalidEvidenceInputError) as exc:
            raise AnswerProviderResponseError("Invalid answer response") from exc
        from financial_agent.observability.recorder import active_recorder
        recorder = active_recorder()
        if recorder is not None:
            recorder.record_writer(draft, "rewrite" if previous_draft is not None else "write")
        return draft


def _generate_traced(provider, component: str, messages, response_schema):
    from financial_agent.observability.recorder import active_recorder

    recorder = active_recorder()
    model = getattr(provider, "model_name", type(provider).__name__)
    call_index = (
        recorder.record_model_call_started(component, model, messages, response_schema)
        if recorder is not None else 0
    )
    try:
        response = provider.generate(messages, response_schema=response_schema)
    except BaseException as exc:
        if recorder is not None:
            recorder.record_model_call_failed(component, call_index, model, exc)
        raise
    if recorder is not None:
        recorder.record_model_call_completed(component, call_index, model, response)
    return response


def _normalize_evidence_paths(raw, results: list[TaskExecutionResult]):
    """Repair common envelope/index path notation, then use normal validation."""
    if not isinstance(raw, dict) or not isinstance(raw.get("evidence"), list):
        return raw
    normalized = deepcopy(raw)
    result_data = {
        result.task_id: _normalization_data(result.result.data)
        for result in results
        if result.result.status == "success"
    }
    for reference in normalized["evidence"]:
        if not isinstance(reference, dict):
            continue
        referenced_item_index = None
        claimed_task_id = reference.get("task_id")
        if claimed_task_id not in result_data and isinstance(claimed_task_id, str):
            matches = [
                (task_id, index)
                for task_id, data in result_data.items()
                if isinstance(data, list)
                for index, item in enumerate(data)
                if isinstance(item, dict)
                and claimed_task_id in {item.get("document_id"), item.get("chunk_id")}
            ]
            if len(matches) == 1:
                reference["task_id"], referenced_item_index = matches[0]
        path = reference.get("source_path")
        if not isinstance(path, list):
            continue
        expanded_path = []
        for segment in path:
            compact_path = _parse_compact_path(segment) if isinstance(segment, str) else None
            expanded_path.extend(compact_path if compact_path is not None else [segment])
        path = expanded_path
        task_data = result_data.get(reference.get("task_id"))
        embedded_path = _embedded_value_path(path, task_data)
        if embedded_path is not None:
            path = embedded_path
        elif (
            path == ["data"]
            and isinstance(task_data, list)
            and task_data
            and isinstance(task_data[0], dict)
            and "content" in task_data[0]
        ):
            path = [0, "content"]
        elif len(path) > 1 and path[0] == "data":
            path = path[1:]
        if (
            path
            and isinstance(path[0], str)
            and isinstance(task_data, list)
            and task_data
            and (
                referenced_item_index is not None
                or (isinstance(task_data[0], dict) and path[0] in task_data[0])
            )
        ):
            path = [referenced_item_index or 0, *path]
        reference["source_path"] = [
            int(segment[1:-1])
            if isinstance(segment, str)
            and len(segment) >= 3
            and segment[0] == "["
            and segment[-1] == "]"
            and segment[1:-1].isdigit()
            and (segment[1:-1] == "0" or not segment[1:-1].startswith("0"))
            else segment
            for segment in path
        ]
    return normalized


def _normalization_data(value):
    """Project typed tool output to the public shape used for path repair."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_normalization_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalization_data(item) for key, item in value.items()}
    return value


def _embedded_value_path(path, task_data) -> list[str | int] | None:
    """Resolve a model-emitted serialized result value back to its unique path.

    Some structured-output models put the complete JSON object in ``source_path``
    instead of emitting its index/key path. Only an exact, unique match within the
    selected task result is repaired; ambiguous or approximate matches still fail
    normal evidence validation.
    """
    if (
        len(path) != 2
        or path[0] != "data"
        or not isinstance(path[1], str)
        or not path[1].lstrip().startswith(("{", "["))
    ):
        return None
    try:
        claimed_value = json.loads(path[1])
    except (json.JSONDecodeError, TypeError):
        return None
    matches = _find_value_paths(task_data, claimed_value)
    return matches[0] if len(matches) == 1 else None


def _find_value_paths(value, target, path=()) -> list[list[str | int]]:
    matches = [list(path)] if value == target else []
    if isinstance(value, dict):
        for key, item in value.items():
            matches.extend(_find_value_paths(item, target, (*path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(_find_value_paths(item, target, (*path, index)))
    return matches


_COMPACT_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[(?:0|[1-9]\d*)\])*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[(?:0|[1-9]\d*)\])*)*$"
)
_COMPACT_PATH_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(0|[1-9]\d*)\]")


def _parse_compact_path(segment: str) -> list[str | int] | None:
    """Parse a strictly limited ``data.items[0].value`` path notation."""
    if "." not in segment and "[" not in segment:
        return None
    if _COMPACT_PATH_RE.fullmatch(segment) is None:
        return None
    return [field if field else int(index) for field, index in _COMPACT_PATH_TOKEN_RE.findall(segment)]
