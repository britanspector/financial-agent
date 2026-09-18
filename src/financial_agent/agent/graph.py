"""Dependency-aware LangGraph execution driven by caller-supplied tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from financial_agent.agent.models import (
    AgentError,
    AgentState,
    FinalResult,
    Task,
    TaskExecutionResult,
)
from financial_agent.agent.retry import ExecutionBudget, RetryPolicy, ToolAttemptBudget
from financial_agent.agent.result_path import ResultPathError, resolve_result_path
from financial_agent.schemas import UserQuery
from financial_agent.tools.contracts import ToolError, ToolResult
from financial_agent.user_data.auth import CallContext

if TYPE_CHECKING:
    from financial_agent.observability.recorder import AgentTraceRecorder, TraceScope


class ToolInvoker(Protocol):
    def input_model(self, name: str): ...
    def invoke(
        self,
        name: str,
        arguments: object,
        *,
        context: CallContext,
        request_id: UUID | None = None,
    ) -> ToolResult: ...


@dataclass(frozen=True)
class ExecutionRuntime:
    """Dependencies and mutable budget isolated to one graph invocation."""

    call_context: CallContext
    retry_policy: RetryPolicy
    budget: ExecutionBudget
    sleeper: Callable[[float], None]
    recorder: "AgentTraceRecorder | None" = None
    trace_scope: "TraceScope | None" = None


def build_execution_graph(
    registry: ToolInvoker,
    *,
    context: CallContext | None = None,
    retry_policy: RetryPolicy | None = None,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
):
    """Compile a graph whose plan is supplied as structured tasks."""
    call_context = context or CallContext()
    policy = retry_policy or RetryPolicy()
    fallback_runtimes: dict[UUID, ExecutionRuntime] = {}
    fallback_lock = Lock()

    def execution_runtime(
        request_id: UUID,
        runtime: Runtime[ExecutionRuntime],
    ) -> ExecutionRuntime:
        if runtime.context is not None:
            return runtime.context
        # Preserve direct use of the compiled graph while keeping its budget
        # isolated by request ID. run_execution_graph always supplies context.
        with fallback_lock:
            if request_id not in fallback_runtimes:
                fallback_runtimes[request_id] = ExecutionRuntime(
                    call_context=call_context,
                    retry_policy=policy,
                    budget=ExecutionBudget(policy, clock=clock),
                    sleeper=sleeper,
                )
            return fallback_runtimes[request_id]

    def dispatcher(
        raw_state: AgentState | Mapping[str, Any],
        runtime: Runtime[ExecutionRuntime],
    ) -> dict[str, Any]:
        state = _state(raw_state)
        run = execution_runtime(state.request_id, runtime)
        completed = {item.task_id: item for item in state.tool_results}
        pending = [task for task in state.tasks if task.task_id not in completed]
        update: dict[str, Any] = {
            "iteration_count": state.iteration_count + 1,
            "scheduled_tasks": [],
        }
        if not pending:
            update["dispatch_action"] = "finalize"
            return update

        failed_ids = {
            task_id for task_id, item in completed.items() if item.result.status == "error"
        }
        task_ids = {task.task_id for task in state.tasks}
        generated: list[TaskExecutionResult] = []

        blocked = [task for task in pending if failed_ids.intersection(task.dependencies)]
        for task in blocked:
            generated.append(_control_error(
                task,
                code="DEPENDENCY_FAILED",
                message="Task blocked because a dependency failed",
                max_retry=run.retry_policy.max_retry,
            ))

        blocked_ids = {item.task_id for item in generated}
        remaining = [task for task in pending if task.task_id not in blocked_ids]
        missing = [
            task for task in remaining if any(dependency not in task_ids for dependency in task.dependencies)
        ]
        for task in missing:
            absent = sorted(set(task.dependencies) - task_ids)
            generated.append(_control_error(
                task,
                code="UNRESOLVED_DEPENDENCY",
                message=f"missing_dependency: {', '.join(absent)}",
                max_retry=run.retry_policy.max_retry,
            ))

        generated_ids = {item.task_id for item in generated}
        remaining = [task for task in remaining if task.task_id not in generated_ids]
        successful_ids = {
            task_id for task_id, item in completed.items() if item.result.status != "error"
        }
        ready = [task for task in remaining if set(task.dependencies).issubset(successful_ids)]
        resolved_ready: list[Task] = []
        for task in ready:
            resolved, error = _resolve_bound_task(task, completed, registry)
            if error is not None:
                generated.append(_control_error(
                    task,
                    code=error[0],
                    message=error[1],
                    max_retry=run.retry_policy.max_retry,
                ))
            else:
                resolved_ready.append(resolved)

        if generated:
            update["tool_results"] = generated
        if resolved_ready:
            update["scheduled_tasks"] = resolved_ready
            update["dispatch_action"] = "execute"
        elif generated:
            update["dispatch_action"] = "collect"
        elif remaining:
            update["tool_results"] = [
                _control_error(
                    task,
                    code="UNRESOLVED_DEPENDENCY",
                    message="cycle_or_deadlock: no dependency-ready task remains",
                    max_retry=run.retry_policy.max_retry,
                )
                for task in remaining
            ]
            update["dispatch_action"] = "collect"
        else:
            update["dispatch_action"] = "collect"
        return update

    def route_dispatch(raw_state: AgentState | Mapping[str, Any]):
        state = _state(raw_state)
        if state.dispatch_action == "execute":
            return [
                Send("execute_tool", {"task": task, "request_id": state.request_id})
                for task in state.scheduled_tasks
            ]
        if state.dispatch_action == "collect":
            return "collect_results"
        return "finalize"

    def execute_tool(
        payload: Mapping[str, Any],
        runtime: Runtime[ExecutionRuntime],
    ) -> dict[str, list[TaskExecutionResult]]:
        task = Task.model_validate(payload["task"])
        run = execution_runtime(UUID(str(payload["request_id"])), runtime)
        attempt_count = 0
        result: ToolResult | None = None
        while True:
            budget_before = run.recorder.projector.budget(run.budget) if run.recorder else None
            if not run.budget.reserve_attempt():
                if run.recorder:
                    _record_in_runtime_scope(
                        run, run.recorder.record_tool_blocked, task, attempt_count + 1,
                        "deadline" if run.budget.deadline_exceeded else "attempt_budget",
                        run.recorder.projector.budget(run.budget),
                    )
                if result is None:
                    task_result = _control_error(
                        task,
                        code="EXECUTION_BUDGET_EXCEEDED",
                        message="Execution budget prevented the Tool attempt from starting",
                        max_retry=run.retry_policy.max_retry,
                    )
                else:
                    task_result = TaskExecutionResult(
                        task_id=task.task_id,
                        tool_name=task.tool_name,
                        result=result,
                        retry_count=max(0, attempt_count - 1),
                        max_retry=run.retry_policy.max_retry,
                    )
                return {"tool_results": [task_result]}

            started = monotonic()
            try:
                result = registry.invoke(
                    task.tool_name,
                    task.arguments,
                    context=run.call_context,
                    request_id=uuid4(),
                )
            except BaseException as exc:
                if run.recorder:
                    _record_in_runtime_scope(run, run.recorder.record_component_failed, "tool", exc)
                raise
            attempt_count += 1
            if run.recorder:
                _record_in_runtime_scope(
                    run, run.recorder.record_tool_attempt, task, attempt_count, result,
                    max(0.0, (monotonic() - started) * 1000), budget_before,
                    run.recorder.projector.budget(run.budget),
                )
            if (
                result.status != "error"
                or result.error is None
                or not result.error.retryable
                or attempt_count >= run.retry_policy.max_retry + 1
            ):
                if run.recorder and result.status == "error":
                    reason = "not_retryable" if result.error is None or not result.error.retryable else "max_retry"
                    _record_in_runtime_scope(
                        run, run.recorder.record_retry, task, attempt_count + 1, 0.0,
                        run.recorder.projector.budget(run.budget), reason=reason,
                    )
                break

            delay = run.retry_policy.backoff_seconds(attempt_count - 1)
            if not run.budget.allows_retry_after(delay):
                if run.recorder:
                    reason = "deadline" if run.budget.deadline_exceeded else "attempt_budget"
                    _record_in_runtime_scope(
                        run, run.recorder.record_retry, task, attempt_count + 1, delay,
                        run.recorder.projector.budget(run.budget), reason=reason,
                    )
                break
            if run.recorder:
                _record_in_runtime_scope(
                    run, run.recorder.record_retry, task, attempt_count + 1, delay,
                    run.recorder.projector.budget(run.budget),
                )
            run.sleeper(delay)

        return {
            "tool_results": [TaskExecutionResult(
                task_id=task.task_id,
                tool_name=task.tool_name,
                result=result,
                retry_count=max(0, attempt_count - 1),
                max_retry=run.retry_policy.max_retry,
            )],
        }

    def collect_results(raw_state: AgentState | Mapping[str, Any]) -> dict[str, Any]:
        state = _state(raw_state)
        new_results = state.tool_results[state.collected_result_count:]
        new_errors = [error for item in new_results if (error := _agent_error(item)) is not None]
        update: dict[str, Any] = {"collected_result_count": len(state.tool_results)}
        if new_errors:
            update["errors"] = new_errors
        return update

    def finalize(
        raw_state: AgentState | Mapping[str, Any],
        runtime: Runtime[ExecutionRuntime],
    ) -> dict[str, Any]:
        state = _state(raw_state)
        run = execution_runtime(state.request_id, runtime)
        results_by_id = {item.task_id: item for item in state.tool_results}
        errors_by_id = {item.task_id: item for item in state.errors}
        ordered_results = [results_by_id[task.task_id] for task in state.tasks]
        ordered_errors = [errors_by_id[task.task_id] for task in state.tasks if task.task_id in errors_by_id]
        if not ordered_errors:
            final_status = "success"
        elif len(ordered_errors) == len(ordered_results):
            final_status = "failed"
        else:
            final_status = "partial"
        final = FinalResult(
            status=final_status,
            query=state.query,
            task_results=ordered_results,
            errors=ordered_errors,
            iteration_count=state.iteration_count,
            attempt_count=run.budget.attempt_count,
            execution_duration_ms=run.budget.elapsed_ms(),
            attempt_budget_exhausted=run.budget.attempt_budget_exhausted,
            deadline_exceeded=run.budget.deadline_exceeded,
        )
        with fallback_lock:
            fallback_runtimes.pop(state.request_id, None)
        return {
            "final_output": final,
            "status": "failed" if final_status == "failed" else "completed",
        }

    builder = StateGraph(AgentState, context_schema=ExecutionRuntime)
    builder.add_node("dispatcher", dispatcher)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("collect_results", collect_results)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "dispatcher")
    builder.add_conditional_edges("dispatcher", route_dispatch)
    builder.add_edge("execute_tool", "collect_results")
    builder.add_edge("collect_results", "dispatcher")
    builder.add_edge("finalize", END)
    return builder.compile()


def run_execution_graph(
    request: UserQuery,
    tasks: Sequence[Task],
    registry: ToolInvoker,
    *,
    context: CallContext | None = None,
    max_concurrency: int | None = None,
    retry_policy: RetryPolicy | None = None,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
    initial_results: Sequence[TaskExecutionResult] | None = None,
    attempt_budget: ToolAttemptBudget | None = None,
) -> AgentState:
    """Execute a caller-supplied task graph and return its validated final state."""
    task_list = list(tasks)
    seeded = list(initial_results or [])
    _validate_initial_results(task_list, seeded)
    state = AgentState.from_query(request, task_list, tool_results=seeded)
    policy = retry_policy or RetryPolicy()
    from financial_agent.observability.recorder import active_recorder, active_scope
    recorder = active_recorder()
    runtime = ExecutionRuntime(
        call_context=context or CallContext(),
        retry_policy=policy,
        budget=ExecutionBudget(policy, clock=clock, attempt_budget=attempt_budget),
        sleeper=sleeper,
        recorder=recorder,
        trace_scope=active_scope() if recorder else None,
    )
    config: dict[str, Any] = {"recursion_limit": max(25, len(tasks) * 4 + 5)}
    if max_concurrency is not None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        config["max_concurrency"] = max_concurrency
    result = build_execution_graph(
        registry,
        context=context,
        retry_policy=policy,
        clock=clock,
        sleeper=sleeper,
    ).invoke(state, config=config, context=runtime)
    return AgentState.model_validate(result)


def _record_in_runtime_scope(run: ExecutionRuntime, callback, *args, **kwargs) -> None:
    """Restore loop correlation explicitly inside graph worker threads."""
    if run.recorder is None:
        return
    scope = run.trace_scope
    with run.recorder.scope(
        operation=scope.operation if scope else "execute",
        iteration=scope.iteration if scope else None,
        plan_revision=scope.plan_revision if scope else None,
    ):
        callback(*args, **kwargs)


def _validate_initial_results(
    tasks: Sequence[Task],
    results: Sequence[TaskExecutionResult],
) -> None:
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Task IDs must be unique")
    result_ids = [item.task_id for item in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Initial result task IDs must be unique")
    for item in results:
        task = task_by_id.get(item.task_id)
        if task is None:
            raise ValueError("Initial results must belong to the current plan")
        if item.tool_name != task.tool_name:
            raise ValueError("Initial result Tool names must match the current plan")
        if item.result.status == "error":
            raise ValueError("Initial reusable results must be success or empty")


def _state(raw_state: AgentState | Mapping[str, Any]) -> AgentState:
    return raw_state if isinstance(raw_state, AgentState) else AgentState.model_validate(raw_state)


def _control_error(
    task: Task,
    *,
    code: str,
    message: str,
    max_retry: int = 0,
) -> TaskExecutionResult:
    return TaskExecutionResult(
        task_id=task.task_id,
        tool_name=task.tool_name,
        result=ToolResult(
            status="error",
            data=None,
            source="agent_control_plane",
            latency=0,
            error=ToolError(
                code=code,
                message=message,
                http_status=424,
                retryable=False,
            ),
            request_id=uuid4(),
        ),
        max_retry=max_retry,
    )


def _agent_error(item: TaskExecutionResult) -> AgentError | None:
    error = item.result.error
    if error is None:
        return None
    reason = None
    if error.code == "DEPENDENCY_FAILED":
        reason = "dependency_failed"
    elif error.code == "UNRESOLVED_DEPENDENCY":
        reason = "missing_dependency" if error.message.startswith("missing_dependency:") else "cycle_or_deadlock"
    return AgentError(
        task_id=item.task_id,
        code=error.code,
        message=error.message,
        http_status=error.http_status,
        retryable=error.retryable,
        reason=reason,
    )


def _resolve_bound_task(
    task: Task,
    completed: Mapping[str, TaskExecutionResult],
    registry: ToolInvoker,
) -> tuple[Task, tuple[str, str] | None]:
    """Resolve simple structured refs only after all dependency results exist."""
    if not task.bindings:
        return task, None
    arguments = dict(task.arguments)
    for binding in task.bindings:
        upstream = completed.get(binding.source_task_id)
        if upstream is None or upstream.result.status == "error":
            return task, ("BINDING_RESOLUTION_FAILED", "Binding source did not complete successfully")
        try:
            value = resolve_result_path(upstream.result.data, binding.source_path)
        except ResultPathError:
            return task, ("BINDING_RESOLUTION_FAILED", "Binding source value was unavailable at runtime")
        arguments[binding.target_parameter] = value
    model = registry.input_model(task.tool_name)
    if model is None:
        return task, None  # preserve Phase 2 unknown-tool result behavior
    try:
        arguments = model.model_validate(arguments).model_dump(mode="json")
    except Exception:
        return task, ("INVALID_BOUND_ARGUMENTS", "Resolved binding values do not satisfy the Tool input schema")
    return task.model_copy(update={"arguments": arguments}), None
