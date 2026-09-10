"""Prevent local environment configuration from affecting tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path, request):
    is_live = request.node.get_closest_marker("live") is not None
    for key in os.environ:
        upper_key = key.upper()
        if is_live and (
            upper_key == "FINANCIAL_AGENT_TUSHARE_TOKEN"
            or upper_key.startswith("FINANCIAL_AGENT_QWEN_")
        ):
            continue
        if upper_key.startswith("FINANCIAL_AGENT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    if not is_live:
        monkeypatch.chdir(tmp_path)
