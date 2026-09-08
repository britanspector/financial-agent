import json
import os
import subprocess
import sys


def run(*args, env=None):
    return subprocess.run([sys.executable, "-m", "financial_agent", *args], capture_output=True, text=True,
                          timeout=30, env=env)


def test_seed_and_tool_cli_end_to_end(tmp_path):
    result = run("seed-user-data")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data/user_data.db").exists()
    assert json.loads(result.stdout)["users"] == 6
    again = run("seed-user-data")
    assert again.returncode == 1 and "--overwrite" in again.stdout
    assert run("seed-user-data", "--overwrite").returncode == 0
    assert len(json.loads(run("list-tools").stdout)) == 3
    secret = "synthetic-cli-test-key"
    env = dict(os.environ, FINANCIAL_AGENT_CALLER_API_KEY=secret, FINANCIAL_AGENT_USER_API_KEYS=json.dumps([
        {"api_key": secret, "principal_id": "synthetic-cli", "user_ids": ["syn-user-001", "syn-user-002"],
         "scopes": ["read:profile", "read:portfolio"]}]))
    for name in ["get_user_profile", "get_user_portfolio", "get_user_transactions"]:
        result = run("call-tool", name, "--user-id", "syn-user-001", env=env)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "success"
        assert secret not in result.stdout + result.stderr
    empty = run("call-tool", "get_user_portfolio", "--user-id", "syn-user-002", env=env)
    assert empty.returncode == 0 and json.loads(empty.stdout)["status"] == "empty"
    denied = run("call-tool", "get_user_profile", "--user-id", "syn-user-001")
    assert denied.returncode == 1 and json.loads(denied.stdout)["error"]["http_status"] == 401
    invalid = run("call-tool", "get_user_transactions", "--user-id", "syn-user-001", "--limit", "0", env=env)
    assert invalid.returncode == 1 and json.loads(invalid.stdout)["error"]["http_status"] == 422


def test_cli_invalid_credentials_do_not_leak():
    env = dict(os.environ, FINANCIAL_AGENT_USER_API_KEYS="synthetic-invalid-secret")
    result = run("call-tool", "get_user_profile", "--user-id", "syn-user-001", env=env)
    assert result.returncode == 1
    assert "synthetic-invalid-secret" not in result.stdout + result.stderr


def test_seed_custom_path(tmp_path):
    path = tmp_path / "custom directory" / "fixtures.db"
    result = run("seed-user-data", "--path", str(path))
    assert result.returncode == 0 and path.exists()
