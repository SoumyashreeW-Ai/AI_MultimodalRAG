from __future__ import annotations

import re
from typing import Iterable

from app.models.document_models import TextChunk


def _chunk_transcript_segment(
    document_id: str,
    filename: str,
    source_path: str,
    page: int,
    text: str,
    start: float,
    end: float,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Chunk a single transcript segment into word-based chunks, interpolating timestamps.

    The timestamps for a chunk are approximated by linear interpolation across words.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= chunk_size:
        return [
            TextChunk(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page,
                chunk_id=f"{document_id}:page-{page}:chunk-1",
                content=normalized,
                content_type="audio",
                start_time=start,
                end_time=end,
            )
        ]

    total_words = len(words)
    duration = max(1e-6, end - start)
    per_word = duration / total_words

    chunks: list[TextChunk] = []
    start_word = 0
    chunk_number = 1
    while start_word < total_words:
        end_word = min(start_word + chunk_size, total_words)
        segment_words = words[start_word:end_word]
        segment_text = " ".join(segment_words)

        seg_start_time = start + start_word * per_word
        seg_end_time = start + end_word * per_word

        chunks.append(
            TextChunk(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page,
                chunk_id=f"{document_id}:page-{page}:chunk-{chunk_number}",
                content=segment_text,
                content_type="audio",
                start_time=seg_start_time,
                end_time=seg_end_time,
            )
        )

        if end_word == total_words:
            break

        start_word += chunk_size - chunk_overlap
        chunk_number += 1

    return chunks


def chunk_transcript_segments(
    document_id: str,
    filename: str,
    source_path: str,
    segments: Iterable[TextChunk] | Iterable[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Chunk multiple transcript segments (each with start/end) into TextChunks.

    Accepts either `TextChunk`-like objects with `content`, `start_time`, `end_time`, or dicts
    from the transcribe step (keys: text,start,end).
    """
    all_chunks: list[TextChunk] = []
    for idx, seg in enumerate(segments, start=1):
        if isinstance(seg, TextChunk):
            text = seg.content
            start = seg.start_time or 0.0
            end = seg.end_time or 0.0
            page = seg.page
        else:
            text = seg.get("text") or ""
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            page = idx

        all_chunks.extend(
            _chunk_transcript_segment(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page,
                text=text,
                start=start,
                end=end,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return all_chunks



def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")


def chunk_text(
    document_id: str,
    filename: str,
    source_path: str,
    page: int,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Chunk one page of text into overlapping windows."""
    _validate_chunk_config(chunk_size, chunk_overlap)

    normalized = _normalize_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= chunk_size:
        return [
            TextChunk(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page,
                chunk_id=f"{document_id}:page-{page}:chunk-1",
                content=normalized,
                content_type="text",
            )
        ]

    chunks: list[TextChunk] = []
    start = 0
    chunk_number = 1
    while start < len(words):
        end = min(start + chunk_size, len(words))
        segment = " ".join(words[start:end])
        if not segment:
            break

        chunks.append(
            TextChunk(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page,
                chunk_id=f"{document_id}:page-{page}:chunk-{chunk_number}",
                content=segment,
                content_type="text",
            )
        )

        if end == len(words):
            break

        start += chunk_size - chunk_overlap
        chunk_number += 1

    return chunks


def chunk_document_pages(
    document_id: str,
    filename: str,
    source_path: str,
    pages: Iterable[tuple[int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Chunk text across multiple pages while preserving page metadata."""
    all_chunks: list[TextChunk] = []
    for page_number, page_text in pages:
        all_chunks.extend(
            chunk_text(
                document_id=document_id,
                filename=filename,
                source_path=source_path,
                page=page_number,
                text=page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks
