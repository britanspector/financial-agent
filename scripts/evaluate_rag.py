"""Run the separated retrieval evaluation set against configured RAG tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_agent.config import Settings
from financial_agent.knowledge.evaluation import RetrievalEvalCase, evaluate_retrieval
from financial_agent.knowledge.runtime import build_rag_tools
from financial_agent.user_data.auth import CallContext


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the persisted RAG index")
    parser.add_argument("--cases", type=Path, default=Path("tests/knowledge/eval_cases.json"))
    args = parser.parse_args()
    cases = [
        RetrievalEvalCase.model_validate(item)
        for item in json.loads(args.cases.read_text(encoding="utf-8"))
    ]
    registry = build_rag_tools(Settings())

    def search(case: RetrievalEvalCase):
        result = registry.invoke(case.tool, case.arguments, context=CallContext())
        if result.status == "error":
            raise RuntimeError(result.error.code)
        return result.data

    metrics = evaluate_retrieval(cases, search)
    print(metrics.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
