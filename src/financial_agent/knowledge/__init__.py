"""Local knowledge corpus ingestion contracts."""

from financial_agent.knowledge.ingestion import KnowledgeIngestor, load_manifest
from financial_agent.knowledge.models import Chunk, Evidence, KnowledgeManifest

__all__ = ["Chunk", "Evidence", "KnowledgeIngestor", "KnowledgeManifest", "load_manifest"]
