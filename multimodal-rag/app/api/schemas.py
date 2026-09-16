from __future__ import annotations

from pydantic import BaseModel
from typing import Any, List, Optional


class HealthResponse(BaseModel):
    status: str
    ok: bool


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class DocumentsListResponse(BaseModel):
    count: int
    documents: Optional[List[Any]] = None


class ReindexRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryResultItem(BaseModel):
    id: str
    score: float
    content_type: str
    document_id: Optional[str]
    filename: Optional[str]
    page: Optional[int]
    chunk_id: Optional[str]
    text: Optional[str]
    image_path: Optional[str]
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Any]
    images: List[str]
    retrieval_debug: Optional[dict] = None
