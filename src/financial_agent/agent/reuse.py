"""Conservative dependency-safe reuse of results across replacement plans."""

from collections.abc import Sequence

from financial_agent.agent.models import Task, TaskExecutionResult


def reusable_results(
    previous_plan: Sequence[Task],
    previous_results: Sequence[TaskExecutionResult],
    new_plan: Sequence[Task],
    *,
    force_rerun_task_ids: set[str] | None = None,
) -> list[TaskExecutionResult]:
    forced = force_rerun_task_ids or set()
    old_tasks = {task.task_id: task for task in previous_plan}
    old_results = {item.task_id: item for item in previous_results}
    candidates = {
        task.task_id
        for task in new_plan
        if task.task_id not in forced
        and old_tasks.get(task.task_id) == task
        and task.task_id in old_results
        and old_results[task.task_id].result.status in {"success", "empty"}
    }
    reusable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for task in new_plan:
            if task.task_id in candidates and task.task_id not in reusable and set(task.dependencies).issubset(reusable):
                reusable.add(task.task_id)
                changed = True
    return [old_results[task.task_id] for task in new_plan if task.task_id in reusable]
