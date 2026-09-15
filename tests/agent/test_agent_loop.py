from uuid import uuid4

from financial_agent.agent.loop import LoopPolicy, run_agent_loop
from financial_agent.agent.models import TaskExecutionResult
from financial_agent.agent.retry import RetryPolicy
from financial_agent.planner.models import PlannedTask, ReplanOutput, StructuredPlan
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Schema, UserQuery
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.verifier.models import DraftAnswer, VerificationResult


class Input(Schema):
    value: int


class Output(Schema):
    value: int


class Registry:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = list(statuses or [])

    def describe(self):
        return [{"name": "lookup", "description": "lookup", "input_schema": Input.model_json_schema()}]

    def input_model(self, name):
        return Input if name == "lookup" else None

    def output_model(self, name):
        return Output if name == "lookup" else None

    def invoke(self, name, arguments, *, context, request_id=None):
        self.calls.append((name, dict(arguments), request_id))
        status = self.statuses.pop(0) if self.statuses else "success"
        error = ToolError(code="TEMP", message="temporary", http_status=503, retryable=True) if status == "error" else None
        return ToolResult(status=status, data=None if error else {"value": arguments["value"]}, source="test",
                          latency=0, error=error, request_id=request_id or uuid4())


def pt(task_id, value, dependencies=None):
    return PlannedTask(task_id=task_id, tool_name="lookup", arguments={"value": value},
                       dependencies=dependencies or [], bindings=[])


class Planner:
    def __init__(self, initial, replans=None):
        self.initial = initial
        self.replans = list(replans or [])
        self.replan_calls = 0

    def plan(self, request):
        return self.initial

    def replan(self, request, plan, results, feedback):
        self.replan_calls += 1
        return self.replans.pop(0)


class Writer:
    def __init__(self):
        self.write_calls = 0
        self.rewrite_calls = 0

    def write(self, request, plan, results):
        self.write_calls += 1
        return DraftAnswer(answer=f"draft-{self.write_calls}")

    def rewrite(self, request, plan, results, draft, feedback):
        self.rewrite_calls += 1
        return DraftAnswer(answer=f"rewrite-{self.rewrite_calls}")


class Verifier:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def verify(self, request, plan, results, draft):
        decision = self.decisions.pop(0)
        return VerificationResult(decision=decision, reason=decision,
                                  missing_evidence=["new fact"] if decision == "REPLAN" else [],
                                  failed_task_ids=[])


def run(initial_tasks, replans, decisions, *, registry=None, policy=None, retry=None):
    registry = registry or Registry()
    planner = Planner(StructuredPlan(decision="execute", tasks=initial_tasks), replans)
    writer = Writer()
    result = run_agent_loop(UserQuery(query="q"), planner, PlanValidator(registry), registry, writer,
                            Verifier(decisions), policy=policy, retry_policy=retry,
                            sleeper=lambda _: None)
    return result, registry, planner, writer


def test_replan_reuses_success_and_drops_results_not_in_replacement_plan():
    replacement = ReplanOutput(tasks=[pt("b", 2), pt("c", 3)], force_rerun_task_ids=[])
    result, registry, _, _ = run([pt("a", 1), pt("b", 2)], [replacement], ["REPLAN", "PASS"])

    assert [call[1]["value"] for call in registry.calls] == [1, 2, 3]
    assert [item.task_id for item in result.task_results] == ["b", "c"]
    assert result.trace[-1].reused_task_ids == ["b"]
    assert result.total_tool_attempts == 3


def test_empty_is_reused_unless_force_rerun_is_explicit():
    registry = Registry(statuses=["empty", "success"])
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=["a"])
    result, registry, _, _ = run([pt("a", 1)], [replacement], ["REPLAN", "PASS"], registry=registry)

    assert len(registry.calls) == 2
    assert result.trace[-1].reused_task_ids == []


def test_empty_is_reused_by_default():
    registry = Registry(statuses=["empty"])
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=[])
    result, registry, _, _ = run([pt("a", 1)], [replacement], ["REPLAN"], registry=registry)
    assert result.stop_reason == "no_progress"
    assert len(registry.calls) == 1


def test_forced_dependency_invalidates_downstream_reuse():
    registry = Registry()
    initial = [pt("a", 1), pt("b", 2, dependencies=["a"])]
    replacement = ReplanOutput(tasks=initial, force_rerun_task_ids=["a"])
    result, registry, _, _ = run(initial, [replacement], ["REPLAN", "PASS"], registry=registry)
    assert len(registry.calls) == 4
    assert result.trace[-1].reused_task_ids == []


def test_error_is_never_reused():
    registry = Registry(statuses=["error", "success"])
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=[])
    result, registry, _, _ = run([pt("a", 1)], [replacement], ["REPLAN", "PASS"], registry=registry,
                                 retry=RetryPolicy(max_retry=0))
    assert len(registry.calls) == 2
    assert result.status == "completed"


def test_tool_budget_does_not_block_rewrite():
    result, registry, planner, writer = run(
        [pt("a", 1)], [], ["REWRITE", "PASS"],
        policy=LoopPolicy(total_tool_budget=1),
    )
    assert result.status == "completed"
    assert result.rewrite_count == 1
    assert writer.rewrite_calls == 1
    assert planner.replan_calls == 0
    assert len(registry.calls) == 1


def test_exhausted_tool_budget_blocks_only_replan():
    replacement = ReplanOutput(tasks=[pt("b", 2)], force_rerun_task_ids=[])
    result, _, planner, _ = run([pt("a", 1)], [replacement], ["REPLAN"],
                                policy=LoopPolicy(total_tool_budget=1))
    assert result.status == "limit_exhausted"
    assert result.stop_reason == "total_tool_budget"
    assert planner.replan_calls == 1
    assert [task.task_id for task in result.plan] == ["a"]


def test_exhausted_tool_budget_allows_replan_that_needs_no_tool():
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=[])
    result, registry, planner, _ = run([pt("a", 1)], [replacement], ["REPLAN"],
                                       policy=LoopPolicy(total_tool_budget=1))
    assert result.stop_reason == "no_progress"
    assert planner.replan_calls == 1
    assert len(registry.calls) == 1


def test_identical_plan_results_and_zero_attempts_is_no_progress():
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=[])
    result, registry, _, writer = run([pt("a", 1)], [replacement], ["REPLAN"])
    assert result.stop_reason == "no_progress"
    assert len(registry.calls) == 1
    assert writer.write_calls == 1


def test_changed_plan_with_zero_attempts_is_not_no_progress():
    replacement = ReplanOutput(tasks=[pt("b", 2)], force_rerun_task_ids=[])
    # b is already present and reusable; removing a changes both plan and result pool.
    result, registry, _, _ = run([pt("a", 1), pt("b", 2)], [replacement], ["REPLAN", "PASS"])
    assert result.status == "completed"
    assert result.stop_reason == "pass"
    assert len(registry.calls) == 2


def test_shared_budget_counts_retry_attempts_across_rounds():
    registry = Registry(statuses=["error", "success", "success"])
    replacement = ReplanOutput(tasks=[pt("a", 1)], force_rerun_task_ids=["a"])
    result, registry, _, _ = run(
        [pt("a", 1)], [replacement], ["REPLAN", "PASS"], registry=registry,
        policy=LoopPolicy(total_tool_budget=3), retry=RetryPolicy(max_retry=1),
    )
    assert len(registry.calls) == 3
    assert result.total_tool_attempts == 3
    assert result.trace[0].new_tool_attempts == 2
    assert result.trace[1].new_tool_attempts == 1


def test_changed_task_definition_is_not_reused():
    replacement = ReplanOutput(tasks=[pt("a", 2)], force_rerun_task_ids=[])
    result, registry, _, _ = run([pt("a", 1)], [replacement], ["REPLAN", "PASS"])
    assert [call[1]["value"] for call in registry.calls] == [1, 2]
    assert result.trace[-1].reused_task_ids == []


def test_rewrite_replan_and_iteration_limits_are_hard():
    rewrite, _, _, _ = run([pt("a", 1)], [], ["REWRITE"],
                            policy=LoopPolicy(max_rewrite=0))
    replan, _, planner, _ = run([pt("a", 1)], [], ["REPLAN"],
                                policy=LoopPolicy(max_replan=0))
    iteration, _, _, _ = run([pt("a", 1)], [], ["REWRITE"],
                              policy=LoopPolicy(max_iterations=1))
    assert rewrite.stop_reason == "max_rewrite"
    assert replan.stop_reason == "max_replan"
    assert planner.replan_calls == 0
    assert iteration.stop_reason == "max_iterations"


def test_clarify_short_circuits_without_tools_or_writer():
    registry = Registry()
    planner = Planner(StructuredPlan(decision="clarify", tasks=[]))
    writer = Writer()
    result = run_agent_loop(UserQuery(query="q"), planner, PlanValidator(registry), registry, writer, Verifier([]))
    assert result.status == "clarify"
    assert result.stop_reason == "clarify"
    assert writer.write_calls == 0
