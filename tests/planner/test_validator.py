import pytest

from financial_agent.planner.models import PlannedTask, StructuredPlan
from financial_agent.planner.validator import PlanValidator

from .conftest import planner_catalog


def plan(*tasks):
    return StructuredPlan(decision="execute", tasks=list(tasks))


def task(task_id="t1", tool="get_customer_context", arguments=None, dependencies=None):
    return PlannedTask(
        task_id=task_id,
        tool_name=tool,
        arguments=arguments if arguments is not None else {"user_id": "syn-user-0001"},
        dependencies=dependencies or [],
    )


def codes(result):
    return [issue.code for issue in result.issues]


def test_valid_plan_normalizes_arguments_for_phase2_task():
    result = PlanValidator(planner_catalog()).validate(plan(task(
        tool="get_margin_account",
        arguments={"user_id": "syn-user-0001", "start_date": "2026-01-01", "end_date": "2026-02-01"},
    )))
    assert result.valid is True
    assert result.tasks[0].arguments["start_date"] == "2026-01-01"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (plan(), "EMPTY_PLAN"),
        (plan(task(tool="not_real")), "UNKNOWN_TOOL"),
        (plan(task(arguments={})), "INVALID_ARGUMENTS"),
        (plan(task(dependencies=["absent"])), "MISSING_DEPENDENCY"),
        (plan(task(dependencies=["t1"])), "SELF_DEPENDENCY"),
        (plan(task(dependencies=["x", "x"])), "DUPLICATE_DEPENDENCY"),
        (plan(task("t1"), task("t1")), "DUPLICATE_TASK_ID"),
        (plan(task("t1", dependencies=["t2"]), task("t2", dependencies=["t1"])), "DEPENDENCY_CYCLE"),
        (plan(task(arguments={"user_id": "$t0.result.user_id"})), "DYNAMIC_RESULT_REFERENCE"),
    ],
)
def test_validator_rejects_each_invalid_plan_class(candidate, expected):
    result = PlanValidator(planner_catalog()).validate(candidate)
    assert result.valid is False
    assert expected in codes(result)
    assert result.tasks == []


def test_task_limit_is_configurable():
    result = PlanValidator(planner_catalog(), max_tasks=1).validate(plan(task("t1"), task("t2")))
    assert "TOO_MANY_TASKS" in codes(result)


@pytest.mark.parametrize("decision", ["clarify", "no_tool"])
def test_validator_accepts_empty_non_execution_decisions(decision):
    result = PlanValidator(planner_catalog()).validate(StructuredPlan(decision=decision, tasks=[]))
    assert result.valid is True
    assert result.decision == decision
    assert result.tasks == []


def test_validator_rejects_tasks_for_non_execution_decision():
    result = PlanValidator(planner_catalog()).validate(StructuredPlan(
        decision="clarify", tasks=[task()],
    ))
    assert result.valid is False
    assert "NON_EXECUTION_HAS_TASKS" in codes(result)


def test_dependencies_only_control_order_and_literal_arguments_are_allowed():
    result = PlanValidator(planner_catalog()).validate(plan(
        task("profile"),
        task("faq", tool="search_business_knowledge", arguments={"query": "融资规则"}, dependencies=["profile"]),
    ))
    assert result.valid is True
    assert result.tasks[1].dependencies == ["profile"]
