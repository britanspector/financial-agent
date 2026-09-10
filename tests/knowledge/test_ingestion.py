import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_agent.knowledge import KnowledgeIngestor, KnowledgeManifest, load_manifest
from financial_agent.knowledge.ingestion import _policy_sections, _split_long_sections


CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"


def test_manifest_has_expected_valid_documents():
    manifest = load_manifest(MANIFEST_PATH)
    counts = Counter(document.metadata.source_type for document in manifest.documents)

    assert len(manifest.documents) == 72
    assert counts == {"research_report": 24, "faq": 24, "announcement_policy": 24}
    assert len({document.document_id for document in manifest.documents}) == 72


def test_corpus_has_eight_markdown_tables_and_no_test_labels():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = ("archetype", "difficulty", "expected_answer", "retrieval_case", "is_latest")
    table_documents = 0
    for entry in payload["documents"]:
        body = (CORPUS_ROOT / entry["path"]).read_text(encoding="utf-8")
        table_documents += "| 指标 | 前期判断 | 本期判断 | 变化说明 |" in body

    assert table_documents == 8
    assert not any(label in serialized for label in forbidden)


def test_ingestion_preserves_metadata_and_uses_one_chunk_per_faq():
    chunks = KnowledgeIngestor().ingest(MANIFEST_PATH)
    counts = Counter(chunk.source_type for chunk in chunks)

    assert len(chunks) == 96
    assert counts["faq"] == 24
    assert counts == {"research_report": 48, "faq": 24, "announcement_policy": 24}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.document_id == chunk.metadata.document_id for chunk in chunks)
    assert all(chunk.source_type == chunk.metadata.source_type for chunk in chunks)


def test_manifest_rejects_duplicate_document_ids():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["documents"].append(payload["documents"][0])

    with pytest.raises(ValidationError, match="globally unique"):
        KnowledgeManifest.model_validate(payload)


def test_manifest_rejects_path_outside_corpus(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["documents"][0]["path"] = "../outside.md"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="leaves knowledge root"):
        load_manifest(tmp_path / "manifest.json")


def test_long_prose_splits_but_markdown_table_stays_atomic():
    long_prose = "长段落。" * 200
    table = "## 指标\n\n| 指标 | 数值 |\n| --- | ---: |\n" + "\n".join(f"| 项目{i} | {i} |" for i in range(80))

    parts = _split_long_sections([long_prose, table], 300)

    assert len(parts) > 2
    assert sum("| 指标 | 数值 |" in part for part in parts) == 1
    assert next(part for part in parts if "| 指标 | 数值 |" in part) == table


def test_policy_chunking_recognizes_article_boundaries():
    sections = _policy_sections("# 规则\n\n第一条 总则。\n第二条 适用范围。")

    assert sections == ["# 规则", "第一条 总则。", "第二条 适用范围。"]
