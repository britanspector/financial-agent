"""Persistent embedding index with deterministic stale detection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from financial_agent.knowledge.dense import DenseIndex
from financial_agent.knowledge.models import Chunk
from financial_agent.knowledge.providers import EmbeddingDescriptor, EmbeddingProvider
from financial_agent.schemas import Schema


class EmbeddingIndexMetadata(Schema):
    schema_version: str = "0.1"
    model: str
    dimension: int
    corpus_fingerprint: str
    chunk_ids: list[str]


class StaleEmbeddingIndexError(ValueError):
    pass


class EmbeddingIndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata_path = path.with_suffix(path.suffix + ".json")

    def state(self, chunks: list[Chunk], descriptor: EmbeddingDescriptor) -> str:
        if not self.path.is_file() or not self.metadata_path.is_file():
            return "missing"
        try:
            metadata = self._read_metadata()
        except (OSError, ValueError, json.JSONDecodeError):
            return "stale"
        expected_ids = [chunk.chunk_id for chunk in chunks]
        if (
            metadata.model != descriptor.model
            or metadata.dimension != descriptor.dimension
            or metadata.chunk_ids != expected_ids
            or metadata.corpus_fingerprint != corpus_fingerprint(chunks)
        ):
            return "stale"
        return "ready"

    def build(self, chunks: list[Chunk], provider: EmbeddingProvider, *, batch_size: int = 20) -> DenseIndex:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batches: list[np.ndarray] = []
        texts = [_embedding_text(chunk) for chunk in chunks]
        for start in range(0, len(texts), batch_size):
            batch = np.asarray(provider.embed_documents(texts[start:start + batch_size]), dtype=np.float32)
            expected_rows = min(batch_size, len(texts) - start)
            if batch.shape != (expected_rows, provider.descriptor.dimension):
                raise ValueError("embedding provider returned an unexpected matrix shape")
            batches.append(batch)
        vectors = np.vstack(batches) if batches else np.empty((0, provider.descriptor.dimension), dtype=np.float32)
        metadata = EmbeddingIndexMetadata(
            model=provider.descriptor.model,
            dimension=provider.descriptor.dimension,
            corpus_fingerprint=corpus_fingerprint(chunks),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        self._save(vectors, metadata)
        return DenseIndex(metadata.chunk_ids, vectors)

    def load(self, chunks: list[Chunk], descriptor: EmbeddingDescriptor) -> DenseIndex:
        state = self.state(chunks, descriptor)
        if state != "ready":
            raise StaleEmbeddingIndexError(f"embedding index is {state}; rebuild it")
        metadata = self._read_metadata()
        try:
            with np.load(self.path, allow_pickle=False) as payload:
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise StaleEmbeddingIndexError("embedding index data is invalid; rebuild it") from exc
        if vectors.shape != (len(metadata.chunk_ids), metadata.dimension):
            raise StaleEmbeddingIndexError("embedding index shape is stale; rebuild it")
        return DenseIndex(metadata.chunk_ids, vectors)

    def _read_metadata(self) -> EmbeddingIndexMetadata:
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return EmbeddingIndexMetadata.model_validate(payload)

    def _save(self, vectors: np.ndarray, metadata: EmbeddingIndexMetadata) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vector_temp = self.path.with_suffix(self.path.suffix + ".tmp")
        metadata_temp = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        try:
            with vector_temp.open("wb") as stream:
                np.savez_compressed(stream, vectors=vectors)
            metadata_temp.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
            os.replace(vector_temp, self.path)
            os.replace(metadata_temp, self.metadata_path)
        finally:
            vector_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.model_dump_json(exclude_none=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _embedding_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.content}"
