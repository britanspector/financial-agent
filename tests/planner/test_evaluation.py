from pathlib import Path

from financial_agent.planner.evaluation import (
    PlannerEvalCase,
    evaluate_planner,
    evaluate_planner_with_report,
    load_planner_eval_cases,
)
from financial_agent.planner.service import StructuredPlanner
from financial_agent.planner.validator import PlanValidator

from .conftest import planner_catalog


CASE_PATH = Path(__file__).parents[2] / "eval" / "planner" / "planner_cases.jsonl"


class CaseFakePlannerProvider:
    def __init__(self, outputs):
        self._outputs = iter(outputs)

    def generate(self, messages, *, response_schema):
        del messages, response_schema
        return next(self._outputs)


def _output_for(case: PlannerEvalCase, *, alternative: bool = False) -> dict:
    pattern = case.acceptable_plans[0] if alternative else case.patterns()[0]
    ids_by_tool = {tool_name: f"generated_{index}" for index, tool_name in enumerate(pattern.expected_tools, 1)}
    arguments = {item.tool_name: item.arguments for item in pattern.expected_arguments}
    dependencies = {tool_name: [] for tool_name in pattern.expected_tools}
    for edge in pattern.expected_dependencies:
        dependencies[edge.downstream_tool].append(ids_by_tool[edge.upstream_tool])
    return {"decision": "no_tool" if case.expectation == "abstain" else "execute", "tasks": [
        {
            "task_id": ids_by_tool[tool_name],
            "tool_name": tool_name,
            "arguments": arguments.get(tool_name, {}),
            "dependencies": dependencies[tool_name],
        }
        for tool_name in pattern.expected_tools
    ]}


def test_fixed_eval_set_has_40_independent_cases_and_required_coverage():
    cases = load_planner_eval_cases(CASE_PATH)

    assert len(cases) == 40
    assert len({case.case_id for case in cases}) == 40
    assert sum(case.expectation == "abstain" for case in cases) == 4
    assert any(len(case.acceptable_plans) for case in cases)
    assert any(len(case.history) for case in cases)
    assert any(item.arguments.get("as_of") for case in cases for item in case.temporal_expectation)
    assert any(item.arguments.get("start_date") for case in cases for item in case.temporal_expectation)
    assert any(item.arguments.get("end_date") for case in cases for item in case.temporal_expectation)
    assert any(len(item.arguments.get("companies", [])) > 1 for case in cases for item in case.expected_arguments)
    assert all("$" not in str(case.model_dump()) for case in cases)


def test_eval_scores_generated_task_ids_semantically_and_accepts_alternative_pattern():
    cases = load_planner_eval_cases(CASE_PATH)
    outputs = [
        _output_for(case, alternative=case.case_id == "acceptable_dependency_variants")
        for case in cases
    ]
    catalog = planner_catalog()

    metrics = evaluate_planner(
        cases,
        StructuredPlanner(CaseFakePlannerProvider(outputs), catalog),
        PlanValidator(catalog),
    )

    assert metrics.case_count == 40
    assert metrics.executable_case_count == 36
    assert metrics.abstention_case_count == 4
    assert metrics.valid_plan_rate == 1
    assert metrics.tool_selection_accuracy == 1
    assert metrics.argument_accuracy == 1
    assert metrics.temporal_accuracy == 1
    assert metrics.dependency_accuracy == 1
    assert metrics.unnecessary_tool_call_rate == 0


def test_forbidden_tool_is_scored_as_unnecessary():
    case = load_planner_eval_cases(CASE_PATH)[0]
    output = _output_for(case)
    output["tasks"].append({
        "task_id": "unrelated",
        "tool_name": "get_market_snapshot",
        "arguments": {"symbol": "600519.SH"},
        "dependencies": [],
    })
    catalog = planner_catalog()

    metrics = evaluate_planner(
        [case], StructuredPlanner(CaseFakePlannerProvider([output]), catalog), PlanValidator(catalog),
    )

    assert metrics.tool_selection_accuracy == 0
    assert metrics.unnecessary_tool_call_rate == 0.5


def test_argument_scoring_normalizes_optional_defaults_but_not_required_values():
    case = PlannerEvalCase.model_validate({
        "case_id": "optional-defaults", "query": "研报", "history": [],
        "expected_tools": ["search_research_reports"],
        "expected_arguments": [{"tool_name": "search_research_reports", "arguments": {"query": "华泰科技"}}],
        "expected_dependencies": [], "temporal_expectation": [], "forbidden_tools": [],
    })
    catalog = planner_catalog()
    provider = CaseFakePlannerProvider([{"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "search_research_reports",
        "arguments": {"query": "华泰科技", "companies": [], "brokers": [], "as_of": None},
        "dependencies": [],
    }]}])
    report = evaluate_planner_with_report([case], StructuredPlanner(provider, catalog), PlanValidator(catalog))

    assert report.metrics.argument_accuracy == 1
    assert report.cases[0].argument_details[0].correct is True

    provider = CaseFakePlannerProvider([{"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "search_research_reports",
        "arguments": {"query": "星河能源", "companies": [], "brokers": [], "as_of": None},
        "dependencies": [],
    }]}])
    report = evaluate_planner_with_report([case], StructuredPlanner(provider, catalog), PlanValidator(catalog))
    assert report.metrics.argument_accuracy == 0


def test_argument_scoring_accepts_query_preserved_entity_but_rejects_wrong_entity_or_category():
    case = PlannerEvalCase.model_validate({
        "case_id": "query-entity", "query": "研报", "history": [],
        "expected_tools": ["search_research_reports"],
        "expected_arguments": [{"tool_name": "search_research_reports", "arguments": {
            "query": "澄海智造经营展望研报", "companies": ["澄海智造"],
        }}],
        "expected_dependencies": [], "temporal_expectation": [], "forbidden_tools": [],
    })
    catalog = planner_catalog()
    provider = CaseFakePlannerProvider([{"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "search_research_reports",
        "arguments": {"query": "检索澄海智造的经营展望研报"}, "dependencies": [],
    }]}])
    report = evaluate_planner_with_report([case], StructuredPlanner(provider, catalog), PlanValidator(catalog))
    assert report.metrics.argument_accuracy == 1
    assert report.cases[0].argument_details[0].status == "acceptable"

    provider = CaseFakePlannerProvider([{"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "search_research_reports",
        "arguments": {"query": "检索澄海智造的经营展望研报", "companies": ["青禾医药"]}, "dependencies": [],
    }]}])
    report = evaluate_planner_with_report([case], StructuredPlanner(provider, catalog), PlanValidator(catalog))
    assert report.metrics.argument_accuracy == 0

    category_case = case.model_copy(update={
        "expected_tools": ["search_business_knowledge"],
        "expected_arguments": [{"tool_name": "search_business_knowledge", "arguments": {
            "query": "市价委托成交", "category": "trading",
        }}],
    })
    provider = CaseFakePlannerProvider([{"decision": "execute", "tasks": [{
        "task_id": "t1", "tool_name": "search_business_knowledge",
        "arguments": {"query": "市价委托成交", "category": "risk_notice"}, "dependencies": [],
    }]}])
    report = evaluate_planner_with_report([category_case], StructuredPlanner(provider, catalog), PlanValidator(catalog))
    assert report.metrics.argument_accuracy == 0
