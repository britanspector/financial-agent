"""Bounded Planner → Execute → Write → Verify control loop."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep
from typing import Literal

from pydantic import Field

from financial_agent.agent.graph import ToolInvoker, run_execution_graph
from financial_agent.agent.models import Task, TaskExecutionResult
from financial_agent.agent.retry import RetryPolicy, ToolAttemptBudget
from financial_agent.agent.reuse import reusable_results
from financial_agent.answering.service import AnswerWriter
from financial_agent.config import Settings
from financial_agent.planner.models import ReplanOutput, StructuredPlan
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Schema, UserQuery
from financial_agent.user_data.auth import CallContext
from financial_agent.verifier.models import DraftAnswer, VerificationResult
from financial_agent.verifier.service import StructuredVerifier


class LoopPolicy(Schema):
    max_rewrite: int = Field(default=2, ge=0, le=100)
    max_replan: int = Field(default=2, ge=0, le=100)
    max_iterations: int = Field(default=5, ge=1, le=1_000)
    total_tool_budget: int = Field(default=36, ge=1, le=10_000)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LoopPolicy":
        return cls(max_rewrite=settings.loop_max_rewrite, max_replan=settings.loop_max_replan,
                   max_iterations=settings.loop_max_iterations, total_tool_budget=settings.loop_total_tool_budget)


class LoopTrace(Schema):
    iteration: int = Field(ge=1)
    action: Literal["initial", "rewrite", "replan"]
    decision: Literal["PASS", "REWRITE", "REPLAN"]
    reused_task_ids: list[str] = Field(default_factory=list)
    new_tool_attempts: int = Field(default=0, ge=0)


class AgentLoopResult(Schema):
    status: Literal["completed", "limit_exhausted", "clarify", "no_tool"]
    answer: str | None = None
    draft: DraftAnswer | None = None
    verification: VerificationResult | None = None
    plan: list[Task] = Field(default_factory=list)
    task_results: list[TaskExecutionResult] = Field(default_factory=list)
    rewrite_count: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)
    iteration_count: int = Field(default=0, ge=0)
    total_tool_attempts: int = Field(default=0, ge=0)
    stop_reason: Literal[
        "pass", "clarify", "no_tool", "max_rewrite", "max_replan", "max_iterations",
        "total_tool_budget", "no_progress",
    ]
    trace: list[LoopTrace] = Field(default_factory=list)


def run_agent_loop(
    request: UserQuery,
    planner: StructuredPlanner,
    validator: PlanValidator,
    registry: ToolInvoker,
    writer: AnswerWriter,
    verifier: StructuredVerifier,
    *,
    policy: LoopPolicy | None = None,
    retry_policy: RetryPolicy | None = None,
    initial_draft: DraftAnswer | None = None,
    context: CallContext | None = None,
    max_concurrency: int | None = None,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
) -> AgentLoopResult:
    limits = policy or LoopPolicy()
    retry = retry_policy or RetryPolicy()
    attempt_budget = ToolAttemptBudget(limits.total_tool_budget)
    initial = planner.plan(request)
    validation = validator.validate(initial)
    if not validation.valid:
        _raise_invalid_plan(validation)
    if validation.decision != "execute":
        decision = validation.decision or "no_tool"
        return AgentLoopResult(status=decision, stop_reason=decision)

    plan = validation.tasks
    results, attempts = _execute(request, plan, registry, [], attempt_budget, retry, context,
                                 max_concurrency, clock, sleeper)
    draft = initial_draft or writer.write(request, plan, results)
    verification = verifier.verify(request, plan, results, draft)
    iterations, rewrites, replans = 1, 0, 0
    trace = [LoopTrace(iteration=1, action="initial", decision=verification.decision,
                       new_tool_attempts=attempts)]

    while verification.decision != "PASS":
        if iterations >= limits.max_iterations:
            return _stopped("max_iterations", draft, verification, plan, results, rewrites, replans,
                            iterations, attempt_budget, trace)
        if verification.decision == "REWRITE":
            if rewrites >= limits.max_rewrite:
                return _stopped("max_rewrite", draft, verification, plan, results, rewrites, replans,
                                iterations, attempt_budget, trace)
            draft = writer.rewrite(request, plan, results, draft, verification)
            rewrites += 1
            verification = verifier.verify(request, plan, results, draft)
            iterations += 1
            trace.append(LoopTrace(iteration=iterations, action="rewrite", decision=verification.decision))
            continue

        if replans >= limits.max_replan:
            return _stopped("max_replan", draft, verification, plan, results, rewrites, replans,
                            iterations, attempt_budget, trace)
        previous_plan, previous_results = plan, results
        replacement: ReplanOutput = planner.replan(request, plan, results, verification)
        validation = validator.validate(StructuredPlan(decision="execute", tasks=replacement.tasks))
        if not validation.valid:
            _raise_invalid_plan(validation)
        new_plan = validation.tasks
        reused = reusable_results(previous_plan, previous_results, new_plan,
                                  force_rerun_task_ids=set(replacement.force_rerun_task_ids))
        replans += 1
        if attempt_budget.exhausted and len(reused) != len(new_plan):
            return _stopped("total_tool_budget", draft, verification, previous_plan, previous_results,
                            rewrites, replans, iterations, attempt_budget, trace)
        plan = new_plan
        results, attempts = _execute(request, plan, registry, reused, attempt_budget, retry, context,
                                     max_concurrency, clock, sleeper)
        if plan == previous_plan and results == previous_results and attempts == 0:
            return _stopped("no_progress", draft, verification, plan, results, rewrites, replans,
                            iterations, attempt_budget, trace)
        draft = writer.write(request, plan, results)
        verification = verifier.verify(request, plan, results, draft)
        iterations += 1
        trace.append(LoopTrace(iteration=iterations, action="replan", decision=verification.decision,
                               reused_task_ids=[item.task_id for item in reused], new_tool_attempts=attempts))

    return AgentLoopResult(status="completed", answer=draft.answer, draft=draft, verification=verification,
                           plan=plan, task_results=results, rewrite_count=rewrites, replan_count=replans,
                           iteration_count=iterations, total_tool_attempts=attempt_budget.attempt_count,
                           stop_reason="pass", trace=trace)


def _execute(request, plan, registry, reused, budget, retry, context, max_concurrency, clock, sleeper):
    before = budget.attempt_count
    state = run_execution_graph(request, plan, registry, context=context, max_concurrency=max_concurrency,
                                retry_policy=retry, clock=clock, sleeper=sleeper,
                                initial_results=reused, attempt_budget=budget)
    assert state.final_output is not None
    return state.final_output.task_results, budget.attempt_count - before


def _raise_invalid_plan(validation) -> None:
    detail = "; ".join(f"{issue.code}: {issue.message}" for issue in validation.issues)
    from financial_agent.agent.orchestrator import InvalidPlanError
    raise InvalidPlanError(detail)


def _stopped(reason, draft, verification, plan, results, rewrites, replans, iterations, budget, trace):
    return AgentLoopResult(status="limit_exhausted", answer=draft.answer, draft=draft,
                           verification=verification, plan=plan, task_results=results,
                           rewrite_count=rewrites, replan_count=replans, iteration_count=iterations,
                           total_tool_attempts=budget.attempt_count, stop_reason=reason, trace=trace)
