from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceReference:
    filename: str | None = None
    page: int | None = None
    content_type: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    document_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    retrieval_debug: dict[str, Any] | None = None
