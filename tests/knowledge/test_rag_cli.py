import json
import os
import subprocess
import sys


def test_build_rag_index_cli_requires_environment_key():
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith("FINANCIAL_AGENT_")
    }
    result = subprocess.run(
        [sys.executable, "-m", "financial_agent", "build-rag-index"],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "status": "error",
        "message": "Invalid or missing RAG configuration",
    }
