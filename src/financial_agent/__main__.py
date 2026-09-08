"""Offline bootstrap check: python -m financial_agent."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from financial_agent.config import Settings
from financial_agent.logging_config import configure_logging
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.fixtures import seed_user_data
from financial_agent.user_data.runtime import build_user_tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Local synthetic user data tools")
    commands = parser.add_subparsers(dest="command")
    seed = commands.add_parser("seed-user-data", help="Generate fixed synthetic SQLite fixtures")
    seed.add_argument("--path", type=Path)
    seed.add_argument("--overwrite", action="store_true")
    commands.add_parser("list-tools", help="Print registered tools and input JSON schemas")
    call = commands.add_parser("call-tool", help="Invoke an authenticated local tool")
    call.add_argument("name")
    call.add_argument("--user-id", required=True)
    call.add_argument("--start-time")
    call.add_argument("--end-time")
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
        for field in ("start_time", "end_time", "limit", "offset"):
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
