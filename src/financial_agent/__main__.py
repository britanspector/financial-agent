"""Offline bootstrap check: python -m financial_agent."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from financial_agent.config import Settings
from financial_agent.logging_config import configure_logging
from financial_agent.knowledge.providers import ProviderError
from financial_agent.knowledge.runtime import build_rag_index
from financial_agent.user_data.api import create_app
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.fixtures import seed_user_data
from financial_agent.user_data.runtime import build_user_tools
from financial_agent.user_data.synthetic_generator import DEFAULT_SEED, generate_synthetic_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Local synthetic user data tools")
    commands = parser.add_subparsers(dest="command")
    seed = commands.add_parser("seed-user-data", help="Generate fixed synthetic SQLite fixtures")
    seed.add_argument("--path", type=Path)
    seed.add_argument("--overwrite", action="store_true")
    generated = commands.add_parser("generate-synthetic-data", help="Generate 2,000 deterministic synthetic margin users")
    generated.add_argument("--path", type=Path, default=Path("data/synthetic-2000.db"))
    generated.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generated.add_argument("--overwrite", action="store_true")
    commands.add_parser("build-rag-index", help="Build the persistent Qwen embedding index")
    commands.add_parser("list-tools", help="Print registered tools and input JSON schemas")
    serve = commands.add_parser("serve-user-data", help="Start the local FastAPI user-data service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    call = commands.add_parser("call-tool", help="Invoke an authenticated local tool")
    call.add_argument("name")
    call.add_argument("--user-id", required=True)
    call.add_argument("--start-time")
    call.add_argument("--end-time")
    call.add_argument("--start-date")
    call.add_argument("--end-date")
    call.add_argument("--limit", type=int)
    call.add_argument("--offset", type=int)
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValidationError:
        # Do not echo configuration input, which may contain secrets.
        print(json.dumps({"phase": 0, "status": "error", "message": "Invalid configuration; check .env.example"}))
        return 1
    logger = configure_logging(settings.log_level)
    if args.command == "serve-user-data":
        import uvicorn
        uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level=settings.log_level.lower())
        return 0
    if args.command == "seed-user-data":
        try:
            seed_user_data(args.path or settings.user_db_path, overwrite=args.overwrite)
        except FileExistsError:
            print(json.dumps({"status": "error", "message": "Database exists; use --overwrite to replace it"}))
            return 1
        except OSError:
            print(json.dumps({"status": "error", "message": "Cannot create synthetic database"}))
            return 1
        print(json.dumps({"status": "ready", "dataset": "synthetic_user_db", "users": 6}))
        return 0
    if args.command == "generate-synthetic-data":
        try:
            result = generate_synthetic_data(args.path, seed=args.seed, overwrite=args.overwrite)
        except FileExistsError:
            print(json.dumps({"status": "error", "message": "Database exists; use --overwrite to replace it"}))
            return 1
        except OSError:
            print(json.dumps({"status": "error", "message": "Cannot create synthetic database"}))
            return 1
        print(json.dumps({
            "status": "ready", "dataset": "synthetic_margin_user_db", "users": len(result.summaries),
            "seed": result.seed, "path": str(result.path),
            "representative_users": result.explanations, "constraints": result.constraints,
            "generation_distribution": result.primary_distribution,
        }, ensure_ascii=False))
        return 0
    if args.command == "build-rag-index":
        try:
            chunks, state = build_rag_index(settings)
        except ValueError:
            print(json.dumps({"status": "error", "message": "Invalid or missing RAG configuration"}))
            return 1
        except ProviderError:
            print(json.dumps({"status": "error", "message": "RAG embedding provider request failed"}))
            return 1
        except OSError:
            print(json.dumps({"status": "error", "message": "Cannot write RAG embedding index"}))
            return 1
        print(json.dumps({
            "status": state,
            "chunks": chunks,
            "model": settings.qwen_embedding_model,
            "dimension": settings.qwen_embedding_dimension,
            "path": str(settings.rag_embedding_index_path),
        }))
        return 0
    if args.command in {"list-tools", "call-tool"}:
        try:
            registry = build_user_tools(settings)
        except ValueError:
            print(json.dumps({"status": "error", "message": "Invalid user API-key configuration"}))
            return 1
        if args.command == "list-tools":
            print(json.dumps(registry.describe()))
            return 0
        arguments = {"user_id": args.user_id}
        for field in ("start_time", "end_time", "start_date", "end_date", "limit", "offset"):
            if getattr(args, field) is not None:
                arguments[field] = getattr(args, field)
        result = registry.invoke(args.name, arguments, context=CallContext(api_key=settings.caller_api_key))
        print(result.model_dump_json())
        return 1 if result.status == "error" else 0
    logger.info("Phase 0 bootstrap ready; data_mode=%s", settings.data_mode)
    print(json.dumps({"phase": 0, "status": "ready", "data_mode": settings.data_mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
