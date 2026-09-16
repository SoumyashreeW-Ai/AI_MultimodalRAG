from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    document_id: str
    filename: str
    source_path: str
    page: int
    chunk_id: str
    content: str
    content_type: str = "text"
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class ImageAsset:
    document_id: str
    filename: str
    source_path: str
    page: int
    image_id: str
    image_path: str
    content_type: str
    caption: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class DocumentAsset:
    document_id: str
    filename: str
    source_path: str
    file_type: str
    sha256: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
