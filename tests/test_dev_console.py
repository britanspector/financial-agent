from time import monotonic, sleep

from fastapi.testclient import TestClient

from financial_agent.agent.loop import AgentLoopResult
from financial_agent.config import Settings
from financial_agent.dev_console import create_dev_console_app
from financial_agent.observability.models import AgentTrace


class _FakeRegistry:
    @staticmethod
    def describe():
        return [{"name": "fake_tool"}]


class _FakeRuntime:
    registry = _FakeRegistry()
    rag_enabled = False

    @staticmethod
    def run(job):
        job.result = AgentLoopResult(status="no_tool", stop_reason="no_tool")
        job.trace = AgentTrace(
            trace_id=job.recorder.trace_id,
            request_id=job.request.request_id,
            capture_mode=job.recorder.capture_mode,
            completion_status="completed",
            events=[],
        )


def _app(tmp_path):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>console</h1>", encoding="utf-8")
    settings = Settings(_env_file=None, qwen_api_key="test-key", tushare_token=None)
    return create_dev_console_app(settings, runtime=_FakeRuntime(), frontend_dir=frontend)


def test_environment_is_safe_and_lists_runtime_tools(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.get("/api/environment")

    assert response.status_code == 200
    assert response.json()["qwen_configured"] is True
    assert response.json()["rag_enabled"] is False
    assert response.json()["tools"] == ["fake_tool"]
    assert "test-key" not in response.text


def test_run_can_be_started_and_polled_to_completion(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        started = client.post("/api/runs", json={"query": "hello", "capture_mode": "evaluation"})
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        deadline = monotonic() + 2
        while True:
            snapshot = client.get(f"/api/runs/{run_id}")
            if snapshot.json()["status"] == "completed" or monotonic() >= deadline:
                break
            sleep(0.01)

    assert snapshot.status_code == 200
    assert snapshot.json()["result"]["status"] == "no_tool"
    assert snapshot.json()["trace"]["completion_status"] == "completed"


def test_unknown_run_returns_404(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.get("/api/runs/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
