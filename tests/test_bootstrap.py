import json
import os
import subprocess
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def test_cli_starts_without_credentials():
    result = subprocess.run([sys.executable, "-m", "financial_agent"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"phase": 0, "status": "ready", "data_mode": "synthetic"}
    assert "bootstrap ready" in result.stderr


def test_cli_rejects_invalid_configuration_without_echoing_secrets():
    env = dict(os.environ, FINANCIAL_AGENT_LOG_LEVEL="synthetic-invalid-secret")
    result = subprocess.run([sys.executable, "-m", "financial_agent"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "error"
    assert "synthetic-invalid-secret" not in result.stdout + result.stderr


def test_langgraph_can_compile_and_invoke_offline():
    # Dependency smoke test only; no application graph is implemented in Phase 0.
    class SmokeState(TypedDict):
        ready: bool

    builder = StateGraph(SmokeState)
    builder.add_node("check", lambda state: {"ready": True})
    builder.add_edge(START, "check")
    builder.add_edge("check", END)
    assert builder.compile().invoke({"ready": False}) == {"ready": True}
