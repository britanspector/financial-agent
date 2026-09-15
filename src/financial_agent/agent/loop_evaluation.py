"""Offline black-box evaluation for the bounded Agent Loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from financial_agent.agent.loop import AgentLoopResult, LoopPolicy, run_agent_loop
from financial_agent.agent.retry import RetryPolicy
from financial_agent.planner.models import PlannedTask, ReplanOutput, StructuredPlan
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.verifier.models import DraftAnswer, VerificationResult


class LoopEvalTask(Schema):
    task_id: str
    value: int | None = None
    dependencies: list[str] = Field(default_factory=list)
    binding_source_task_id: str | None = None

    def planned(self) -> PlannedTask:
        bindings = []
        if self.binding_source_task_id is not None:
            bindings = [{
                "target_parameter": "value",
                "source_task_id": self.binding_source_task_id,
                "source_path": ["value"],
            }]
        return PlannedTask(
            task_id=self.task_id,
            tool_name="lookup",
            arguments={} if bindings else {"value": self.value},
            dependencies=self.dependencies,
            bindings=bindings,
        )


class LoopEvalReplan(Schema):
    tasks: list[LoopEvalTask]
    force_rerun_task_ids: list[str] = Field(default_factory=list)


class LoopEvalExpected(Schema):
    status: Literal["completed", "limit_exhausted"]
    stop_reason: str
    iteration_count: int
    rewrite_count: int
    replan_count: int
    total_tool_attempts: int
    final_task_ids: list[str]
    tool_call_values: list[int]
    trace_actions: list[str]
    rewrite_is_tool_free: bool = True
    rewrite_preserves_results: bool = True


class LoopEvalCase(Schema):
    case_id: str
    initial_tasks: list[LoopEvalTask]
    replans: list[LoopEvalReplan] = Field(default_factory=list)
    verifier_decisions: list[Literal["PASS", "REWRITE", "REPLAN"]]
    tool_statuses: list[Literal["success", "empty", "error"]] = Field(default_factory=list)
    max_retry: int = 0
    policy: dict[str, int] = Field(default_factory=dict)
    expected: LoopEvalExpected


class LoopEvalResult(Schema):
    case_id: str
    passed: bool
    failures: list[str]
    actual: AgentLoopResult
    tool_call_values: list[int]
    rewrite_is_tool_free: bool
    rewrite_preserves_results: bool


class LoopEvaluationReport(Schema):
    case_count: int
    passed_count: int
    all_passed: bool
    cases: list[LoopEvalResult]


class _Input(Schema):
    value: int


class _Output(Schema):
    value: int


class _Registry:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.calls: list[int] = []

    def describe(self):
        return [{"name": "lookup", "description": "Return a synthetic integer",
                 "input_schema": _Input.model_json_schema()}]

    def input_model(self, name):
        return _Input if name == "lookup" else None

    def output_model(self, name):
        return _Output if name == "lookup" else None

    def invoke(self, name, arguments, *, context, request_id=None):
        del name, context
        value = arguments["value"]
        self.calls.append(value)
        status = self.statuses.pop(0) if self.statuses else "success"
        error = None
        if status == "error":
            error = ToolError(code="TEMPORARY_FAILURE", message="synthetic failure",
                              http_status=503, retryable=True)
        return ToolResult(
            status=status,
            data=None if error else {"value": value},
            source="synthetic_loop_eval",
            latency=0,
            error=error,
            request_id=request_id or uuid4(),
        )


class _Planner:
    def __init__(self, case: LoopEvalCase) -> None:
        self._initial = StructuredPlan(decision="execute", tasks=[task.planned() for task in case.initial_tasks])
        self._replans = [ReplanOutput(
            tasks=[task.planned() for task in item.tasks],
            force_rerun_task_ids=item.force_rerun_task_ids,
        ) for item in case.replans]

    def plan(self, request):
        del request
        return self._initial

    def replan(self, request, plan, results, feedback):
        del request, plan, results, feedback
        return self._replans.pop(0)


class _Writer:
    def __init__(self, registry: _Registry) -> None:
        self._registry = registry
        self._latest_results = None
        self._latest_tool_count = 0
        self.rewrite_is_tool_free = True
        self.rewrite_preserves_results = True
        self._sequence = 0

    @staticmethod
    def _snapshot(results):
        return [item.model_dump(mode="json") for item in results]

    def write(self, request, plan, results):
        del request, plan
        self._latest_results = self._snapshot(results)
        self._latest_tool_count = len(self._registry.calls)
        self._sequence += 1
        return DraftAnswer(answer=f"draft-{self._sequence}")

    def rewrite(self, request, plan, results, draft, feedback):
        del request, plan, draft, feedback
        before = len(self._registry.calls)
        self.rewrite_preserves_results &= self._snapshot(results) == self._latest_results
        self.rewrite_is_tool_free &= before == self._latest_tool_count
        self._sequence += 1
        after = len(self._registry.calls)
        self.rewrite_is_tool_free &= before == after
        self._latest_results = self._snapshot(results)
        self._latest_tool_count = after
        return DraftAnswer(answer=f"rewrite-{self._sequence}")


class _Verifier:
    def __init__(self, decisions: list[str]) -> None:
        self._decisions = list(decisions)

    def verify(self, request, plan, results, draft):
        del request, plan, results, draft
        decision = self._decisions.pop(0)
        return VerificationResult(
            decision=decision,
            reason=f"synthetic {decision}",
            missing_evidence=["additional synthetic evidence"] if decision == "REPLAN" else [],
        )


def load_loop_eval_cases(path: Path) -> list[LoopEvalCase]:
    cases = [LoopEvalCase.model_validate(json.loads(line))
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Loop eval case IDs must be unique")
    return cases


def evaluate_loop_cases(cases: list[LoopEvalCase]) -> LoopEvaluationReport:
    observations = [_run_case(case) for case in cases]
    return LoopEvaluationReport(
        case_count=len(observations),
        passed_count=sum(item.passed for item in observations),
        all_passed=all(item.passed for item in observations),
        cases=observations,
    )


def _run_case(case: LoopEvalCase) -> LoopEvalResult:
    registry = _Registry(case.tool_statuses)
    writer = _Writer(registry)
    actual = run_agent_loop(
        UserQuery(query=f"loop eval {case.case_id}"),
        _Planner(case),
        PlanValidator(registry),
        registry,
        writer,
        _Verifier(case.verifier_decisions),
        policy=LoopPolicy(**case.policy),
        retry_policy=RetryPolicy(max_retry=case.max_retry),
        sleeper=lambda _: None,
    )
    expected = case.expected
    comparisons = {
        "status": actual.status == expected.status,
        "stop_reason": actual.stop_reason == expected.stop_reason,
        "iteration_count": actual.iteration_count == expected.iteration_count,
        "rewrite_count": actual.rewrite_count == expected.rewrite_count,
        "replan_count": actual.replan_count == expected.replan_count,
        "total_tool_attempts": actual.total_tool_attempts == expected.total_tool_attempts,
        "final_task_ids": [item.task_id for item in actual.task_results] == expected.final_task_ids,
        "tool_call_values": sorted(registry.calls) == sorted(expected.tool_call_values),
        "trace_actions": [item.action for item in actual.trace] == expected.trace_actions,
        "rewrite_is_tool_free": writer.rewrite_is_tool_free == expected.rewrite_is_tool_free,
        "rewrite_preserves_results": writer.rewrite_preserves_results == expected.rewrite_preserves_results,
    }
    failures = [name for name, passed in comparisons.items() if not passed]
    return LoopEvalResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        actual=actual,
        tool_call_values=registry.calls,
        rewrite_is_tool_free=writer.rewrite_is_tool_free,
        rewrite_preserves_results=writer.rewrite_preserves_results,
    )
