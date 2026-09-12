"""Single Phase 3 entry point: plan, compile bindings, execute, return FinalResult."""

from __future__ import annotations

from financial_agent.agent.graph import ToolInvoker, run_execution_graph
from financial_agent.agent.models import FinalResult
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import UserQuery
from financial_agent.user_data.auth import CallContext


class InvalidPlanError(ValueError):
    """Raised before execution when deterministic plan/binding compilation fails."""


def run_planner_execution(
    request: UserQuery,
    planner: StructuredPlanner,
    validator: PlanValidator,
    registry: ToolInvoker,
    *,
    context: CallContext | None = None,
    max_concurrency: int | None = None,
) -> FinalResult:
    """Run the complete query/history → Planner → Validator → graph pipeline."""
    validation = validator.validate(planner.plan(request))
    if not validation.valid:
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in validation.issues)
        raise InvalidPlanError(detail)
    if validation.decision != "execute":
        return FinalResult(status="success", query=request.query, task_results=[], errors=[], iteration_count=0)
    state = run_execution_graph(
        request, validation.tasks, registry, context=context, max_concurrency=max_concurrency,
    )
    assert state.final_output is not None
    return state.final_output
