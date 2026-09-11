"""Prompt-independent, semantic metrics for the fixed planner evaluation set."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from financial_agent.planner.models import PlannedTask, StructuredPlan
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator
from financial_agent.schemas import Message, Schema, UserQuery


class ExpectedToolArguments(Schema):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExpectedDependency(Schema):
    upstream_tool: str = Field(min_length=1)
    downstream_tool: str = Field(min_length=1)


class AcceptablePlanPattern(Schema):
    """One semantically acceptable plan, without model task-id assumptions."""

    expected_tools: list[str] = Field(default_factory=list)
    expected_arguments: list[ExpectedToolArguments] = Field(default_factory=list)
    expected_dependencies: list[ExpectedDependency] = Field(default_factory=list)
    temporal_expectation: list[ExpectedToolArguments] = Field(default_factory=list)


class PlannerEvalCase(Schema):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[Message]
    expected_tools: list[str]
    expected_arguments: list[ExpectedToolArguments]
    expected_dependencies: list[ExpectedDependency]
    temporal_expectation: list[ExpectedToolArguments]
    forbidden_tools: list[str]
    acceptable_plans: list[AcceptablePlanPattern] = Field(default_factory=list)
    expectation: Literal["executable", "abstain"] = "executable"

    def request(self) -> UserQuery:
        return UserQuery(query=self.query, history=self.history)

    def patterns(self) -> list[AcceptablePlanPattern]:
        canonical = AcceptablePlanPattern(
            expected_tools=self.expected_tools,
            expected_arguments=self.expected_arguments,
            expected_dependencies=self.expected_dependencies,
            temporal_expectation=self.temporal_expectation,
        )
        return [canonical, *self.acceptable_plans]


class PlannerEvalMetrics(Schema):
    case_count: int
    executable_case_count: int
    abstention_case_count: int
    valid_plan_rate: float
    tool_selection_accuracy: float
    argument_accuracy: float
    temporal_accuracy: float
    dependency_accuracy: float
    unnecessary_tool_call_rate: float


class ArgumentEvaluation(Schema):
    tool_name: str
    expected: dict[str, Any]
    actual: dict[str, Any] | None = None
    correct: bool
    status: Literal["exact", "acceptable", "incorrect"]
    reason: str


class PlannerCaseEvaluation(Schema):
    case_id: str
    query: str
    expectation: Literal["executable", "abstain"]
    expected_plan: AcceptablePlanPattern
    actual_plan: StructuredPlan
    validation_valid: bool
    validation_issues: list[str]
    tool_correct: bool
    argument_correct: bool
    temporal_correct: bool
    dependency_correct: bool
    temporal_correct_count: int
    temporal_total_count: int
    dependency_score: float
    unnecessary_calls: int
    planned_task_count: int
    argument_details: list[ArgumentEvaluation]
    error_reasons: list[str]


class PlannerEvaluationReport(Schema):
    metrics: PlannerEvalMetrics
    cases: list[PlannerCaseEvaluation]


def load_planner_eval_cases(path: Path) -> list[PlannerEvalCase]:
    """Load newline-delimited independent test specifications with useful errors."""
    cases: list[PlannerEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(PlannerEvalCase.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid planner eval case at line {line_number}") from exc
    if not cases:
        raise ValueError("At least one eval case is required")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Planner eval case IDs must be unique")
    return cases


def evaluate_planner(
    cases: list[PlannerEvalCase],
    planner: StructuredPlanner,
    validator: PlanValidator,
) -> PlannerEvalMetrics:
    return evaluate_planner_with_report(cases, planner, validator).metrics


def evaluate_planner_with_report(
    cases: list[PlannerEvalCase],
    planner: StructuredPlanner,
    validator: PlanValidator,
) -> PlannerEvaluationReport:
    if not cases:
        raise ValueError("At least one eval case is required")
    valid = executable_count = abstention_count = 0
    tool_correct = argument_correct = argument_total = 0
    temporal_correct = temporal_total = 0
    dependency_score = 0.0
    unnecessary = planned_total = 0
    reports: list[PlannerCaseEvaluation] = []

    for case in cases:
        actual = planner.plan(case.request())
        validation = validator.validate(actual)
        if case.expectation == "executable":
            executable_count += 1
            valid += int(validation.valid)
        else:
            abstention_count += 1

        candidates = [_score_pattern(actual, pattern, case.forbidden_tools, validator) for pattern in case.patterns()]
        score = max(candidates, key=lambda candidate: candidate.rank)
        tool_correct += int(score.tools_exact)
        argument_correct += score.argument_correct
        argument_total += score.argument_total
        temporal_correct += score.temporal_correct
        temporal_total += score.temporal_total
        dependency_score += score.dependency_score
        unnecessary += score.unnecessary_calls
        planned_total += len(actual.tasks)

        reports.append(_case_report(case, actual, validation, score))

    metrics = PlannerEvalMetrics(
        case_count=len(cases),
        executable_case_count=executable_count,
        abstention_case_count=abstention_count,
        valid_plan_rate=valid / executable_count if executable_count else 1.0,
        tool_selection_accuracy=tool_correct / len(cases),
        argument_accuracy=argument_correct / argument_total if argument_total else 1.0,
        temporal_accuracy=temporal_correct / temporal_total if temporal_total else 1.0,
        dependency_accuracy=dependency_score / len(cases),
        unnecessary_tool_call_rate=unnecessary / planned_total if planned_total else 0.0,
    )
    return PlannerEvaluationReport(metrics=metrics, cases=reports)


class _PatternScore:
    def __init__(
        self,
        *,
        tools_exact: bool,
        argument_correct: int,
        argument_total: int,
        temporal_correct: int,
        temporal_total: int,
        dependency_score: float,
        unnecessary_calls: int,
        argument_details: list[ArgumentEvaluation],
        expected: AcceptablePlanPattern,
    ) -> None:
        self.tools_exact = tools_exact
        self.argument_correct = argument_correct
        self.argument_total = argument_total
        self.temporal_correct = temporal_correct
        self.temporal_total = temporal_total
        self.dependency_score = dependency_score
        self.unnecessary_calls = unnecessary_calls
        self.argument_details = argument_details
        self.expected = expected

    @property
    def rank(self) -> tuple[float, ...]:
        return (
            float(self.tools_exact),
            self.argument_correct / self.argument_total if self.argument_total else 1.0,
            self.temporal_correct / self.temporal_total if self.temporal_total else 1.0,
            self.dependency_score,
            -float(self.unnecessary_calls),
        )


def _score_pattern(
    actual: StructuredPlan,
    expected: AcceptablePlanPattern,
    forbidden_tools: list[str],
    validator: PlanValidator,
) -> _PatternScore:
    actual_tools = Counter(task.tool_name for task in actual.tasks)
    expected_tools = Counter(expected.expected_tools)
    tools_exact = actual_tools == expected_tools and not (set(actual_tools) & set(forbidden_tools))
    unnecessary = sum((actual_tools - expected_tools).values())

    remaining_arguments = list(actual.tasks)
    argument_correct = 0
    argument_details: list[ArgumentEvaluation] = []
    for item in expected.expected_arguments:
        matching_index = next(
            (index for index, task in enumerate(remaining_arguments)
             if task.tool_name == item.tool_name and _compare_arguments(
                 item.tool_name, item.arguments, task.arguments, validator,
             )[0] > 0),
            None,
        )
        if matching_index is not None:
            task = remaining_arguments.pop(matching_index)
            match_status, match_reason = _compare_arguments(
                item.tool_name, item.arguments, task.arguments, validator,
            )
            argument_correct += 1
            argument_details.append(ArgumentEvaluation(
                tool_name=item.tool_name, expected=item.arguments, actual=task.arguments,
                correct=True,
                status="exact" if match_status == 2 else "acceptable",
                reason=match_reason,
            ))
        else:
            actual_task = next((task for task in actual.tasks if task.tool_name == item.tool_name), None)
            argument_details.append(ArgumentEvaluation(
                tool_name=item.tool_name,
                expected=item.arguments,
                actual=actual_task.arguments if actual_task else None,
                correct=False,
                status="incorrect",
                reason="missing matching Tool instance" if actual_task is None else "required value or non-default argument differs",
            ))

    temporal_correct = 0
    remaining = list(actual.tasks)
    for item in expected.temporal_expectation:
        for index, task in enumerate(remaining):
            if task.tool_name == item.tool_name and _contains_arguments(task.arguments, item.arguments):
                temporal_correct += 1
                remaining.pop(index)
                break

    actual_edges = Counter(_tool_edges(actual.tasks))
    expected_edges = Counter(
        (edge.upstream_tool, edge.downstream_tool) for edge in expected.expected_dependencies
    )
    return _PatternScore(
        tools_exact=tools_exact,
        argument_correct=argument_correct,
        argument_total=len(expected.expected_arguments),
        temporal_correct=temporal_correct,
        temporal_total=len(expected.temporal_expectation),
        dependency_score=_f1(actual_edges, expected_edges),
        unnecessary_calls=unnecessary,
        argument_details=argument_details,
        expected=expected,
    )


_ENTITY_FILTERS = {
    "companies": "companies",
    "brokers": "brokers",
    "issuer": "issuer",
}
_STRICT_ARGUMENTS = {"user_id", "symbol", "as_of", "start_date", "end_date", "category"}


def _compare_arguments(
    tool_name: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    validator: PlanValidator,
) -> tuple[int, str]:
    """Compare Tool inputs with strict critical values and narrow query relaxation.

    Public-schema normalization makes omitted optionals, null and empty lists
    equal where the Tool contract defines the same default.  The only further
    relaxation is query wording and an omitted entity filter already preserved
    verbatim in query; critical values and wrong entity/category/date values
    remain failures.
    """
    expected_normalized = validator.normalize_arguments(tool_name, expected)
    actual_normalized = validator.normalize_arguments(tool_name, actual)
    if expected_normalized is None or actual_normalized is None:
        return 0, "arguments do not validate against the public Tool schema"
    if expected_normalized == actual_normalized:
        return 2, "schema-normalized arguments are equivalent"

    for name in _STRICT_ARGUMENTS:
        if name in expected and expected_normalized.get(name) != actual_normalized.get(name):
            return 0, f"strict argument differs: {name}"

    if "query" in expected and not _queries_equivalent(
        str(expected_normalized.get("query", "")), str(actual_normalized.get("query", "")),
    ):
        return 0, "query is not semantically equivalent"

    relaxed_filter = False
    actual_query = str(actual_normalized.get("query", ""))
    for name, normalized_name in _ENTITY_FILTERS.items():
        expected_value = expected_normalized.get(normalized_name)
        if not expected_value:
            continue
        actual_value = actual_normalized.get(normalized_name)
        if actual_value:
            if actual_value != expected_value:
                return 0, f"entity filter differs: {name}"
            continue
        entities = expected_value if isinstance(expected_value, list) else [expected_value]
        if not all(_preserved_in_query(str(entity), actual_query) for entity in entities):
            return 0, f"missing entity filter not preserved in query: {name}"
        relaxed_filter = True

    ignored = {"query", *_ENTITY_FILTERS}
    if any(
        expected_normalized.get(name) != actual_normalized.get(name)
        for name in set(expected_normalized) | set(actual_normalized)
        if name not in ignored
    ):
        return 0, "non-query argument differs"
    if relaxed_filter:
        return 1, "entity filter omitted but preserved verbatim in query"
    return 1, "query wording is semantically equivalent"


def _queries_equivalent(expected: str, actual: str) -> bool:
    expected_normalized = _normalize_query(expected)
    actual_normalized = _normalize_query(actual)
    return bool(expected_normalized and actual_normalized and (
        expected_normalized in actual_normalized or actual_normalized in expected_normalized
    ))


def _preserved_in_query(entity: str, query: str) -> bool:
    return _normalize_query(entity) in _normalize_query(query)


def _normalize_query(value: str) -> str:
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.lower())
    normalized = re.sub(r"\d{8}", "", normalized)
    for filler in ("查询", "检索", "查看", "查找", "关于", "的", "截至", "当前", "有效", "最新", "历史"):
        normalized = normalized.replace(filler, "")
    return normalized


def _case_report(
    case: PlannerEvalCase,
    actual: StructuredPlan,
    validation: Any,
    score: _PatternScore,
) -> PlannerCaseEvaluation:
    reasons: list[str] = []
    if not validation.valid:
        reasons.extend(f"validator: {issue.code}" for issue in validation.issues)
    if not score.tools_exact:
        reasons.append("tool selection differs from selected expected plan")
    if score.argument_correct != score.argument_total:
        reasons.append("one or more required values or non-default arguments differ")
    if score.temporal_correct != score.temporal_total:
        reasons.append("temporal expectation differs")
    if score.dependency_score != 1:
        reasons.append("dependency edges differ")
    if score.unnecessary_calls:
        reasons.append(f"{score.unnecessary_calls} unnecessary Tool call(s)")
    return PlannerCaseEvaluation(
        case_id=case.case_id,
        query=case.query,
        expectation=case.expectation,
        expected_plan=score.expected,
        actual_plan=actual,
        validation_valid=validation.valid,
        validation_issues=[issue.code for issue in validation.issues],
        tool_correct=score.tools_exact,
        argument_correct=score.argument_correct == score.argument_total,
        temporal_correct=score.temporal_correct == score.temporal_total,
        dependency_correct=score.dependency_score == 1,
        temporal_correct_count=score.temporal_correct,
        temporal_total_count=score.temporal_total,
        dependency_score=score.dependency_score,
        unnecessary_calls=score.unnecessary_calls,
        planned_task_count=len(actual.tasks),
        argument_details=score.argument_details,
        error_reasons=reasons or ["all scored dimensions correct"],
    )


def _tool_edges(tasks: list[PlannedTask]) -> list[tuple[str, str]]:
    names = {task.task_id: task.tool_name for task in tasks}
    return [
        (names.get(dependency, f"<missing:{dependency}>"), task.tool_name)
        for task in tasks
        for dependency in task.dependencies
    ]


def _contains_arguments(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(name) == value for name, value in expected.items())


def _stable_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _f1(actual: Counter[tuple[str, str]], expected: Counter[tuple[str, str]]) -> float:
    if not actual and not expected:
        return 1.0
    matched = sum((actual & expected).values())
    if not matched:
        return 0.0
    precision = matched / sum(actual.values())
    recall = matched / sum(expected.values())
    return 2 * precision * recall / (precision + recall)
