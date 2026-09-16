"""Evidence-grounded structured draft writer."""

from pydantic import ValidationError

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
        raw = self._provider.generate(
            build_answer_messages(
                contextual_request,
                plan,
                results,
                previous_draft=previous_draft,
                feedback=feedback,
                history_summary=selection.summary,
                retrieved_history=selection.retrieved_history,
            ),
            response_schema=answer_response_schema(),
        )
        try:
            draft = DraftAnswer.model_validate(raw)
            resolve_evidence(draft.evidence, plan, results, self._catalog)
        except (ValidationError, InvalidEvidenceInputError) as exc:
            raise AnswerProviderResponseError("Invalid answer response") from exc
        from financial_agent.observability.recorder import active_recorder
        recorder = active_recorder()
        if recorder is not None:
            recorder.record_writer(draft, "rewrite" if previous_draft is not None else "write")
        return draft
