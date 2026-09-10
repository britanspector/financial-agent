"""Tool-facing service boundary for knowledge retrieval."""

from __future__ import annotations

from time import perf_counter
from uuid import UUID, uuid4

from pydantic import ValidationError

from financial_agent.knowledge.models import (
    BusinessSearchInput,
    Evidence,
    EvidenceList,
    RegulatorySearchInput,
    ResearchSearchInput,
)
from financial_agent.knowledge.providers import (
    ProviderError,
    ProviderPermissionDeniedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from financial_agent.knowledge.retrieval import HybridKnowledgeRetriever
from financial_agent.tools.contracts import ToolFailure, ToolResult
from financial_agent.tools.registry import ToolSpec
from financial_agent.user_data.auth import CallContext


class KnowledgeRetrievalService:
    def __init__(self, retriever: HybridKnowledgeRetriever) -> None:
        self._retriever = retriever

    def execute(
        self,
        spec: ToolSpec | None,
        arguments: object,
        *,
        context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult[list[Evidence]]:
        del context
        started = perf_counter()
        request_id = request_id or uuid4()
        evidence: list[Evidence] | None = None
        error = None
        status = "error"
        try:
            if spec is None:
                raise ToolFailure("UNKNOWN_TOOL", "Unknown tool", 404)
            try:
                parameters = spec.input_model.model_validate(arguments)
            except ValidationError:
                raise ToolFailure("INVALID_ARGUMENT", "Invalid knowledge-search arguments", 422) from None
            if isinstance(parameters, ResearchSearchInput):
                evidence = self._retriever.search_research(parameters)
            elif isinstance(parameters, RegulatorySearchInput):
                evidence = self._retriever.search_regulatory(parameters)
            elif isinstance(parameters, BusinessSearchInput):
                evidence = self._retriever.search_business(parameters)
            else:
                raise ToolFailure("INVALID_ARGUMENT", "Invalid knowledge-search arguments", 422)
            evidence = EvidenceList.model_validate(evidence).root
            status = "success" if evidence else "empty"
        except ToolFailure as exc:
            error = exc.error
        except ProviderError as exc:
            error = _provider_failure(exc).error
        except Exception:
            error = ToolFailure("INTERNAL_ERROR", "Internal knowledge-search failure", 500).error
        elapsed = max(0.0, (perf_counter() - started) * 1_000)
        if error is not None:
            evidence, status = None, "error"
        return ToolResult[list[Evidence]](
            status=status,
            data=evidence,
            source="synthetic_knowledge_corpus",
            latency=elapsed,
            error=error,
            request_id=request_id,
        )


def _provider_failure(error: ProviderError) -> ToolFailure:
    operation = "Embedding" if error.operation == "embedding" else "Reranker"
    if isinstance(error, ProviderTimeoutError):
        return ToolFailure("PROVIDER_TIMEOUT", f"{operation} provider timed out", 504, retryable=True)
    if isinstance(error, ProviderUnavailableError):
        return ToolFailure("PROVIDER_UNAVAILABLE", f"{operation} provider unavailable", 503, retryable=True)
    if isinstance(error, ProviderPermissionDeniedError):
        return ToolFailure("PROVIDER_PERMISSION_DENIED", f"{operation} provider rejected credentials", 403)
    if isinstance(error, ProviderResponseError):
        return ToolFailure("PROVIDER_ERROR", f"{operation} provider returned an invalid response", 502)
    return ToolFailure("PROVIDER_ERROR", f"{operation} provider request failed", 502)
