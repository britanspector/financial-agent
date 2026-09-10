import json
import os
import subprocess
import sys
import socket
import time

import pytest


def run(*args, env=None):
    return subprocess.run([sys.executable, "-m", "financial_agent", *args], capture_output=True, text=True, timeout=30, env=env)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
def test_generate_and_tool_cli_end_to_end(tmp_path):
    db = tmp_path / "synthetic-20.db"
    result = run("generate-synthetic-data", "--path", str(db), "--seed", "20260910")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["users"] == 2000
    assert len(json.loads(run("list-tools").stdout)) == 4
    secret = "synthetic-cli-test-key"
    env = dict(os.environ, FINANCIAL_AGENT_USER_DB_PATH=str(db), FINANCIAL_AGENT_CALLER_API_KEY=secret,
               FINANCIAL_AGENT_USER_DATA_BASE_URL="http://127.0.0.1:0",
               FINANCIAL_AGENT_USER_API_KEYS=json.dumps([{
                   "api_key": secret, "principal_id": "synthetic-cli", "user_ids": ["syn-user-0001"],
                   "scopes": ["read:customer_context", "read:margin_account", "read:portfolio_positions", "read:portfolio_analytics"],
               }]))
    port = free_port()
    env["FINANCIAL_AGENT_USER_DATA_BASE_URL"] = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "financial_agent", "serve-user-data", "--port", str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        for name in ["get_customer_context", "get_margin_account", "get_portfolio_positions", "get_portfolio_analytics"]:
            result = run("call-tool", name, "--user-id", "syn-user-0001", env=env)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["status"] == "success"
            assert secret not in result.stdout + result.stderr
        invalid = run("call-tool", "get_margin_account", "--user-id", "syn-user-0001", "--limit", "0", env=env)
        assert invalid.returncode == 1 and json.loads(invalid.stdout)["error"]["http_status"] == 422
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_cli_invalid_credentials_do_not_leak(tmp_path):
    db = tmp_path / "synthetic-20.db"
    run("generate-synthetic-data", "--path", str(db), "--seed", "20260910")
    secret = "synthetic-invalid-secret"
    env = dict(os.environ, FINANCIAL_AGENT_USER_DB_PATH=str(db), FINANCIAL_AGENT_USER_API_KEYS=secret)
    result = run("call-tool", "get_customer_context", "--user-id", "syn-user-0001", env=env)
    assert result.returncode == 1
    assert secret not in result.stdout + result.stderr
