"""Strict manifest and chunk contracts for Multi-source RAG v0.1."""

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, RootModel, model_validator

from financial_agent.schemas import Schema


class ResearchReportMetadata(Schema):
    document_id: str = Field(min_length=1)
    source_type: Literal["research_report"]
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    broker: str = Field(min_length=1)
    publish_date: date
    report_period: str = Field(min_length=1)


class FAQMetadata(Schema):
    document_id: str = Field(min_length=1)
    source_type: Literal["faq"]
    title: str = Field(min_length=1)
    category: Literal["account", "trading", "margin", "product_feature", "risk_notice"]
    version: str = Field(min_length=1)
    effective_date: date


class AnnouncementPolicyMetadata(Schema):
    document_id: str = Field(min_length=1)
    source_type: Literal["announcement_policy"]
    title: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    publish_date: date
    effective_date: date
    status: Literal["active", "superseded", "expired"]


DocumentMetadata = Annotated[
    ResearchReportMetadata | FAQMetadata | AnnouncementPolicyMetadata,
    Field(discriminator="source_type"),
]


class ManifestDocument(Schema):
    document_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def consistent_document_id(self):
        if self.document_id != self.metadata.document_id:
            raise ValueError("manifest document_id must match metadata document_id")
        return self


class KnowledgeManifest(Schema):
    schema_version: Literal["0.1"]
    documents: list[ManifestDocument]

    @model_validator(mode="after")
    def unique_documents(self):
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be globally unique")
        paths = [document.path for document in self.documents]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest paths must be unique")
        return self


class Chunk(Schema):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_type: Literal["research_report", "faq", "announcement_policy"]
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def consistent_metadata(self):
        if self.document_id != self.metadata.document_id:
            raise ValueError("chunk document_id must match metadata document_id")
        if self.source_type != self.metadata.source_type:
            raise ValueError("chunk source_type must match metadata source_type")
        return self


class Evidence(Schema):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_type: Literal["research_report", "faq", "announcement_policy"]
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def consistent_metadata(self):
        if self.document_id != self.metadata.document_id:
            raise ValueError("evidence document_id must match metadata document_id")
        if self.source_type != self.metadata.source_type:
            raise ValueError("evidence source_type must match metadata source_type")
        return self


class EvidenceList(RootModel[list[Evidence]]):
    """Root-list schema used as the shared output contract of all RAG tools."""


class ResearchSearchInput(Schema):
    query: str = Field(min_length=1, max_length=2_000)
    companies: list[str] = Field(default_factory=list, max_length=20)
    brokers: list[str] = Field(default_factory=list, max_length=20)
    as_of: date | None = Field(default=None, description="Inclusive publication-date upper bound")


class RegulatorySearchInput(Schema):
    query: str = Field(min_length=1, max_length=2_000)
    issuer: str | None = Field(default=None, min_length=1, max_length=128)
    as_of: date | None = Field(default=None, description="Inclusive knowledge-time upper bound")


class BusinessSearchInput(Schema):
    query: str = Field(min_length=1, max_length=2_000)
    category: Literal["account", "trading", "margin", "product_feature", "risk_notice"] | None = None
    as_of: date | None = Field(default=None, description="Inclusive effective-date upper bound")
