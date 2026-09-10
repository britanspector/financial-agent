"""Validated configuration; loading is explicit, never performed at import time."""

from typing import Literal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINANCIAL_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    data_mode: Literal["synthetic"] = "synthetic"
    model_api_key: SecretStr | None = Field(default=None, exclude=True)
    user_db_path: Path = Path("data/synthetic-2000.db")
    audit_path: Path = Path("data/audit.jsonl")
    user_data_base_url: str = "http://127.0.0.1:8000"
    user_data_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    user_api_keys: SecretStr = Field(default=SecretStr("[]"), exclude=True, repr=False)
    caller_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    tushare_token: SecretStr | None = Field(default=None, exclude=True, repr=False)
    tushare_base_url: str = "https://api.tushare.pro"
    market_data_timeout_seconds: float = Field(default=10.0, gt=0, le=300)
