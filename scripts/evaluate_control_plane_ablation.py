"""Run the deterministic Phase 6.3 control-plane ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_agent.agent.control_plane_ablation import (
    evaluate_control_plane_ablation, load_control_plane_ablation,
)
from financial_agent.observability import JsonlTraceSink


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline control-plane ablation")
    parser.add_argument("--json-report", type=Path, help="Write the complete typed report")
    parser.add_argument("--trace-jsonl", type=Path, help="Append one evaluation trace per matrix cell")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    eval_set = load_control_plane_ablation(
        root / "eval" / "control_plane_ablation" / "combined_manifest.json",
    )
    report = evaluate_control_plane_ablation(
        eval_set,
        trace_sink=JsonlTraceSink(args.trace_jsonl) if args.trace_jsonl else None,
    )
    print(json.dumps({
        "manifest_hash": report.manifest_hash,
        "case_count": report.case_count,
        "run_count": report.run_count,
        "phase61_full_compatible": report.phase61_full_compatible,
        "recovery_staircase_valid": report.recovery_staircase_valid,
        "all_passed": report.all_passed,
        "metrics": [item.model_dump(mode="json") for item in report.metrics],
        "deltas": [item.model_dump(mode="json") for item in report.deltas],
        "attributions": [item.model_dump(mode="json") for item in report.attributions],
    }, ensure_ascii=False, indent=2))
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
