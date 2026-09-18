import json
from pathlib import Path

from financial_agent.agent.live_evaluation import (
    FrozenLiveConfig, LiveEvalSet, LiveGold, LiveScenario, evaluate_live_agent,
    load_live_eval_set, rescore_live_failure_attribution,
)
from financial_agent.planner.providers import PlannerProviderTimeoutError


ROOT = Path(__file__).parents[2]


class _Provider:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def generate(self, messages, *, response_schema):
        del messages, response_schema
        if self.error:
            raise self.error
        return self.output


def _small_eval_set():
    config = FrozenLiveConfig(
        schema_version="1.0", scenario_path="x", scenario_sha256="0" * 64,
        gold_path="y", gold_sha256="1" * 64, case_count=15,
        configurations=["simple_baseline", "full_agent"],
        execution_order="case_major_simple_then_full",
        model="qwen-test-fixed", temperature=0, max_infra_reruns=1,
    )
    scenario = LiveScenario(
        case_id="mock", query="risk for syn-user-0001", mechanisms=["single_tool"],
    )
    gold = LiveGold.model_validate({
        "case_id": "mock", "solvable": True, "expected_decision": "execute",
        "acceptable_initial_plans": [[{
            "tool_name": "get_customer_context", "arguments": {"user_id": "syn-user-0001"},
        }]],
        "expected_tools": [{
            "tool_name": "get_customer_context", "arguments": {"user_id": "syn-user-0001"},
        }],
        "required_text": ["R3"],
        "required_evidence": [{
            "tool_name": "get_customer_context", "arguments": {"user_id": "syn-user-0001"},
            "source_path": ["risk_level"],
        }],
    })
    return LiveEvalSet(
        manifest_sha256="2" * 64, frozen_config=config, scenarios=[scenario], gold=[gold],
    )


def _factory(component, case_id, configuration, run_attempt):
    del case_id, configuration, run_attempt
    outputs = {
        "planner": {"decision": "execute", "tasks": [{
            "task_id": "customer", "tool_name": "get_customer_context",
            "arguments": {"user_id": "syn-user-0001"}, "dependencies": [], "bindings": [],
        }]},
        "writer": {"answer": "Risk level is R3.", "evidence": [{
            "task_id": "customer", "source_path": ["risk_level"],
        }]},
        "verifier": {"decision": "PASS", "reason": "grounded", "missing_evidence": []},
        "summary": {"facts": []},
    }
    return _Provider(outputs[component])


def test_frozen_live_holdout_loads_with_expected_coverage_and_hashes():
    loaded = load_live_eval_set(ROOT / "eval" / "live_agent_e2e" / "frozen_config.json")
    assert len(loaded.scenarios) == 15
    assert loaded.frozen_config.model == "qwen3.7-flash-2026-07-15"
    assert loaded.frozen_config.temperature == 0
    assert [item.case_id for item in loaded.scenarios] == [item.case_id for item in loaded.gold]


def test_live_runner_scores_both_configurations_from_unified_trace():
    report = evaluate_live_agent(_small_eval_set(), _factory)
    assert report.cell_count == 2
    assert report.run_attempt_count == 2
    assert all(item.task_success_rate == 1 for item in report.aggregates)
    assert all(cell.attempts[0].planner_correct for cell in report.cells)
    assert all(cell.attempts[0].trace_health == "healthy" for cell in report.cells)


def test_provider_timeout_allows_exactly_one_preserved_full_run_retry():
    def factory(component, case_id, configuration, run_attempt):
        if component == "planner" and run_attempt == 1:
            return _Provider(error=PlannerProviderTimeoutError("timeout"))
        return _factory(component, case_id, configuration, run_attempt)

    report = evaluate_live_agent(_small_eval_set(), factory)
    assert report.run_attempt_count == 4
    for cell in report.cells:
        assert len(cell.attempts) == 2
        assert cell.attempts[0].failure.domain == "provider/infra_error"
        assert cell.attempts[0].valid_for_effect_metrics is False
        assert cell.attempts[1].task_success is True
        assert cell.selected_attempt == 2


def test_frozen_config_rejects_configuration_matrix_expansion():
    payload = json.loads((ROOT / "eval" / "live_agent_e2e" / "frozen_config.json").read_text())
    payload["configurations"] = ["full_agent", "simple_baseline"]
    try:
        FrozenLiveConfig.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid live configuration order was accepted")


def test_failure_rescore_changes_invalid_plan_domain_without_model_calls():
    report = evaluate_live_agent(_small_eval_set(), _factory)
    cell = report.cells[0]
    attempt = cell.attempts[0]
    bad = attempt.failure.model_copy(update={
        "domain": "agent_control_error", "code": "InvalidPlanError", "component": "agent_loop",
    })
    replaced = attempt.model_copy(update={
        "failure": bad, "audit": attempt.audit.model_copy(update={"failure": bad}),
    })
    report.cells[0] = cell.model_copy(update={"attempts": [replaced]})
    rescored = rescore_live_failure_attribution(report, _small_eval_set())
    corrected = rescored.cells[0].attempts[0]
    assert corrected.failure.domain == "model_error"
    assert corrected.failure.component == "planner"
    assert rescored.scorer_schema_fixes_after_freeze
