from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    id: str
    score: float
    content_type: str
    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    text: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
