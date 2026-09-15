"""Run the fixed Phase 4.2 Verifier Eval against configured Qwen."""

from __future__ import annotations

import argparse
from pathlib import Path

from financial_agent.config import Settings
from financial_agent.schemas import Schema
from financial_agent.verifier.evaluation import evaluate_verifier, load_verifier_eval_cases
from financial_agent.verifier.runtime import build_verifier


class _EvalOutput(Schema):
    value: str


class _EvalCatalog:
    def output_model(self, name):
        return _EvalOutput if name == "lookup_fact" else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Verifier Eval against configured Qwen")
    parser.add_argument("--json-report", type=Path, help="Write a machine-readable report")
    args = parser.parse_args()
    settings = Settings()
    verifier = build_verifier(settings, _EvalCatalog())
    case_path = Path(__file__).parents[1] / "eval" / "verifier" / "verifier_cases.jsonl"
    report = evaluate_verifier(load_verifier_eval_cases(case_path), verifier)
    print(report.metrics.model_dump_json(indent=2))
    if args.json_report:
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
