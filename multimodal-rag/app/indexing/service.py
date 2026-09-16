from __future__ import annotations

from pathlib import Path
from typing import Any

from app.embeddings.providers import ImageEmbeddingProvider, TextEmbeddingProvider, ImageCaptionProvider
from app.ingestion.chunking import chunk_document_pages
from app.ingestion.parsers import parse_document
from app.vectorstore.collections import ChromaVectorStore
from time import perf_counter
from app.observability.opik import get_opik
import json


class IndexingService:
    """Load, parse, chunk, embed, and store documents in ChromaDB."""

    def __init__(
        self,
        vectorstore: ChromaVectorStore,
        text_embedding_provider: TextEmbeddingProvider,
        image_embedding_provider: ImageEmbeddingProvider,
        caption_provider: ImageCaptionProvider | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.vectorstore = vectorstore
        self.text_embedding_provider = text_embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.caption_provider = caption_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_document(
        self,
        file_path: str | Path,
        document_id: str | None = None,
        storage_dir: str | Path = "data/uploads",
    ) -> dict[str, Any]:
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")

        asset = parse_document(source_path, document_id=document_id, storage_dir=storage_dir)
        duplicate = self.vectorstore.document_exists(asset.document_id, sha256=asset.sha256)
        if duplicate:
            return {
                "document_id": asset.document_id,
                "filename": asset.filename,
                "text_chunks": 0,
                "images": 0,
                "status": "duplicate",
            }

        # If the parsed chunks are audio-transcript segments, chunk them while
        # preserving start/end timestamps rather than naively re-chunking text.
        if any(getattr(chunk, "content_type", "text") == "audio" for chunk in asset.text_chunks):
            # Use transcript-aware chunking to preserve timestamps
            from app.ingestion.chunking import chunk_transcript_segments

            chunked_text = chunk_transcript_segments(
                document_id=asset.document_id,
                filename=asset.filename,
                source_path=asset.source_path,
                segments=asset.text_chunks,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            page_entries = [(chunk.page, chunk.content) for chunk in asset.text_chunks]
            chunked_text = chunk_document_pages(
                document_id=asset.document_id,
                filename=asset.filename,
                source_path=asset.source_path,
                pages=page_entries,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        text_records = []
        text_embeddings = []
        if chunked_text:
            opik = get_opik()
            t0 = perf_counter()
            text_vectors = self.text_embedding_provider.embed_texts([chunk.content for chunk in chunked_text])
            dur = perf_counter() - t0
            if opik:
                opik.log_metric("index.embedding_time_sec", dur)
                opik.log_metric("index.batch_size", len(chunked_text))
                opik.log_text("index.filenames", json.dumps([getattr(c, 'filename', None) for c in chunked_text][:20]))
            for chunk, vector in zip(chunked_text, text_vectors):
                text_records.append(
                    {
                        "id": chunk.chunk_id,
                        "text": chunk.content,
                        "metadata": {
                            "document_id": chunk.document_id,
                            "filename": chunk.filename,
                            "source_path": chunk.source_path,
                            "page": chunk.page,
                            "chunk_id": chunk.chunk_id,
                            "content_type": chunk.content_type,
                            "sha256": asset.sha256,
                            "start_time": getattr(chunk, "start_time", None),
                            "end_time": getattr(chunk, "end_time", None),
                        },
                    }
                )
                text_embeddings.append(vector)
            self.vectorstore.add_text_chunks(text_records, text_embeddings)

        image_records = []
        image_embeddings = []
        if asset.images:
            # Attempt to generate optional captions for images. Failures should not abort indexing.
            if self.caption_provider is not None:
                for image in asset.images:
                    if not getattr(image, "caption", None):
                        try:
                            image.caption = self.caption_provider.caption_image(image.image_path)
                        except Exception:
                            # Ignore caption failures and continue ingestion
                            image.caption = None

            opik = get_opik()
            t0 = perf_counter()
            image_vectors = self.image_embedding_provider.embed_images(
                [image.image_path for image in asset.images]
            )
            dur = perf_counter() - t0
            if opik:
                opik.log_metric("index.image_embedding_time_sec", dur)
                opik.log_metric("index.images_count", len(asset.images))
            for image, vector in zip(asset.images, image_vectors):
                image_records.append(
                    {
                        "id": image.image_id,
                        "caption": image.caption or "",
                        "metadata": {
                            "document_id": image.document_id,
                            "filename": image.filename,
                            "source_path": image.source_path,
                            "page": image.page,
                            "image_id": image.image_id,
                            "image_path": image.image_path,
                            "content_type": image.content_type,
                            "sha256": asset.sha256,
                            "caption": image.caption or "",
                        },
                    }
                )
                image_embeddings.append(vector)
            self.vectorstore.add_images(image_records, image_embeddings)

        return {
            "document_id": asset.document_id,
            "filename": asset.filename,
            "text_chunks": len(chunked_text),
            "images": len(asset.images),
            "status": "indexed",
        }
