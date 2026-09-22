# Financial Agent Developer Console

This is a local developer UI for inspecting a real Financial Agent run. It keeps one
browser-local conversation and shows the complete `evaluation` trace for the latest run.

## Start

From the repository root in PowerShell:

```powershell
python -m financial_agent serve-dev-console --host 127.0.0.1 --port 4181 --model qwen3.7-flash-2026-07-15
```

From Command Prompt (`cmd.exe`), use the same command on its own line:

```bat
python -m financial_agent serve-dev-console --host 127.0.0.1 --port 4181 --model qwen3.7-flash-2026-07-15
```

Then open `http://127.0.0.1:4181`.

The provider adapters ignore the machine-wide HTTP proxy configuration. API keys remain
inside the server process and are never returned to the browser.

## Current UI contract

- `GET /api/environment` returns non-secret runtime metadata.
- `POST /api/runs` accepts `query`, prior `history`, and `capture_mode`.
- `GET /api/runs/{run_id}` returns status, result, and accumulated trace events.
- The conversation is stored in browser `localStorage`; runs remain process-local.
- The trace panel groups events by loop iteration and exposes every Planner, Tool, Writer,
  and Verifier call. Model cards include exact input messages, response schema, and raw
  model output in `evaluation` mode.
- Tool cards include the full arguments, result, status, latency, retry state, and budget.

`safe` traces intentionally omit raw values. The developer UI requests `evaluation` mode
because its purpose is local debugging. Do not bind this server to a public network.
