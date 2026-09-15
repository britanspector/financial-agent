"""Validated configuration; loading is explicit, never performed at import time."""

from typing import Literal
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
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
    knowledge_manifest_path: Path = Path("data/knowledge/manifest.json")
    rag_embedding_index_path: Path = Path("data/knowledge/index/embeddings.npz")
    rag_chunk_max_chars: int = Field(default=400, ge=300, le=8_000)
    rag_rrf_k: int = Field(default=60, ge=1, le=1_000)
    qwen_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    qwen_embedding_model: Literal[
        "qwen3.7-text-embedding", "qwen3.7-text-embedding-flash"
    ] = "qwen3.7-text-embedding"
    qwen_embedding_dimension: int = Field(default=1024)
    qwen_embedding_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    qwen_reranker_model: Literal["qwen3.7-text-rerank"] = "qwen3.7-text-rerank"
    qwen_reranker_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    qwen_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    planner_model: str = Field(default="qwen3.7-flash-2026-07-15", min_length=1)
    planner_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    planner_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    planner_temperature: float = Field(default=0.1, ge=0, le=2)
    planner_max_tasks: int = Field(default=12, ge=1, le=100)
    verifier_model: str = Field(default="qwen3.7-flash", min_length=1)
    verifier_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    verifier_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    verifier_temperature: float = Field(default=0.0, ge=0, le=2)
    answer_model: str = Field(default="qwen3.7-flash", min_length=1)
    answer_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    answer_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    answer_temperature: float = Field(default=0.1, ge=0, le=2)
    context_strategy: Literal[
        "full_history", "last_n", "budgeted_selection", "summary_compression"
    ] = "full_history"
    context_last_n: int = Field(default=6, ge=0, le=100_000)
    context_summary_recent_n: int = Field(default=3, ge=0, le=100_000)
    context_summary_budget_ratio: float = Field(default=0.4, ge=0, le=1)
    context_summary_cache_size: int = Field(default=128, ge=0, le=100_000)
    context_summary_max_facts: int = Field(default=24, ge=1, le=1_000)
    planner_context_budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    answer_context_budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    verifier_context_budget_tokens: int = Field(default=4096, ge=0, le=1_000_000)
    summary_model: str = Field(default="qwen3.7-flash", min_length=1)
    summary_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    summary_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    summary_temperature: float = Field(default=0.0, ge=0, le=2)
    loop_max_rewrite: int = Field(default=2, ge=0, le=100)
    loop_max_replan: int = Field(default=2, ge=0, le=100)
    loop_max_iterations: int = Field(default=5, ge=1, le=1_000)
    loop_total_tool_budget: int = Field(default=36, ge=1, le=10_000)
    execution_max_retry: int = Field(default=2, ge=0, le=100)
    execution_initial_backoff_seconds: float = Field(default=0.5, ge=0, le=300)
    execution_backoff_multiplier: float = Field(default=2.0, ge=1, le=100)
    execution_max_attempts: int = Field(default=36, ge=1, le=10_000)
    execution_deadline_seconds: float = Field(default=120.0, gt=0, le=86_400)
    qwen_embedding_batch_size: int = Field(default=20, ge=1, le=20)
    qwen_embedding_query_instruct: str = "Retrieve relevant passages from a financial knowledge base."
    qwen_reranker_instruct: str = "Given a financial search query, retrieve passages that answer the query."

    @model_validator(mode="after")
    def valid_qwen_embedding_dimension(self):
        full_dimensions = {256, 512, 768, 1024, 1536, 2048, 2560}
        flash_dimensions = {256, 512, 768, 1024}
        allowed = flash_dimensions if self.qwen_embedding_model.endswith("-flash") else full_dimensions
        if self.qwen_embedding_dimension not in allowed:
            raise ValueError("Unsupported dimension for configured Qwen embedding model")
        return self
