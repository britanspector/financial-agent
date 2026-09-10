"""Offline manifest loading and source-aware Markdown chunking."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from financial_agent.knowledge.models import Chunk, KnowledgeManifest, ManifestDocument


def load_manifest(manifest_path: Path) -> KnowledgeManifest:
    """Load a manifest and validate every referenced Markdown document."""
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = KnowledgeManifest.model_validate(payload)
    root = manifest_path.parent
    for document in manifest.documents:
        relative_path = Path(document.path)
        if relative_path.is_absolute():
            raise ValueError(f"document path must be relative: {document.document_id}")
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"document path leaves knowledge root: {document.document_id}")
        if target.suffix.lower() != ".md" or not target.is_file():
            raise ValueError(f"Markdown document not found: {document.document_id}")
        if not target.read_text(encoding="utf-8").strip():
            raise ValueError(f"Markdown document is empty: {document.document_id}")
    return manifest


class KnowledgeIngestor:
    """Turn a validated local corpus into stable, retrieval-ready chunks."""

    def __init__(self, max_chars: int = 400):
        if max_chars < 300:
            raise ValueError("max_chars must be at least 300")
        self._max_chars = max_chars

    def ingest(self, manifest_path: Path) -> list[Chunk]:
        manifest_path = manifest_path.resolve()
        manifest = load_manifest(manifest_path)
        chunks: list[Chunk] = []
        for document in manifest.documents:
            content = (manifest_path.parent / document.path).read_text(encoding="utf-8").strip()
            parts = self._chunk_document(document, content)
            for ordinal, part in enumerate(parts):
                chunks.append(Chunk(
                    chunk_id=f"{document.document_id}::chunk-{ordinal:04d}",
                    document_id=document.document_id,
                    source_type=document.metadata.source_type,
                    title=document.metadata.title,
                    content=part,
                    ordinal=ordinal,
                    metadata=document.metadata,
                ))
        return chunks

    def _chunk_document(self, document: ManifestDocument, content: str) -> list[str]:
        if document.metadata.source_type == "faq":
            return [content]
        if document.metadata.source_type == "research_report":
            sections = _markdown_sections(content)
            blocks = _split_long_sections(sections, self._max_chars)
            return list(_pack_sections(blocks, self._max_chars))
        sections = _policy_sections(content)
        blocks = _split_long_sections(sections, self._max_chars * 2)
        return list(_pack_sections(blocks, self._max_chars * 2))


def _markdown_sections(content: str) -> list[str]:
    """Split only at level-two headings so tables and paragraphs stay together."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.startswith("## ") and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(section).strip() for section in sections if "\n".join(section).strip()]


def _policy_sections(content: str) -> list[str]:
    """Split formal text at headings and explicit Chinese article boundaries."""
    sections: list[list[str]] = []
    current: list[str] = []
    article = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+条(?:\s|　|$)")
    for line in content.splitlines():
        boundary = line.startswith("## ") or article.match(line) is not None
        if boundary and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(section).strip() for section in sections if "\n".join(section).strip()]


def _split_long_sections(sections: list[str], max_chars: int) -> list[str]:
    blocks: list[str] = []
    for section in sections:
        if len(section) <= max_chars or _contains_markdown_table(section):
            blocks.append(section)
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= max_chars or _contains_markdown_table(paragraph):
                blocks.append(paragraph)
                continue
            blocks.extend(_split_sentences(paragraph, max_chars))
    return blocks


def _contains_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    return any(
        line.strip().startswith("|") and line.strip().endswith("|")
        for line in lines
    )


def _split_sentences(text: str, max_chars: int) -> list[str]:
    sentences = [item for item in re.split(r"(?<=[。！？；])", text) if item]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            parts.append(current.strip())
            current = ""
        if len(sentence) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(sentence[index:index + max_chars].strip() for index in range(0, len(sentence), max_chars))
        else:
            current += sentence
    if current.strip():
        parts.append(current.strip())
    return [part for part in parts if part]


def _pack_sections(sections: list[str], max_chars: int) -> Iterable[str]:
    current: list[str] = []
    current_length = 0
    for section in sections:
        added_length = len(section) + (2 if current else 0)
        if current and current_length + added_length > max_chars:
            yield "\n\n".join(current)
            current = []
            current_length = 0
        current.append(section)
        current_length += len(section) + (2 if len(current) > 1 else 0)
    if current:
        yield "\n\n".join(current)
