"""Prevent local environment configuration from affecting tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for key in os.environ:
        if key.upper().startswith("FINANCIAL_AGENT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.chdir(tmp_path)
