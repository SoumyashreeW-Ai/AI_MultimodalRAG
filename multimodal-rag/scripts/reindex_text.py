"""Safely reindex text into a separate Chroma collection.

This development-only command never deletes a collection and never writes to the
image collection. Re-running it upserts stable chunk IDs into the selected new
text collection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.embeddings.providers import build_default_text_embedding_provider
from app.ingestion.chunking import chunk_document_pages
from app.ingestion.parsers import parse_document
from app.vectorstore.collections import ChromaVectorStore


TEXT_DOCUMENT_SUFFIXES = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown"}


def iter_text_documents(uploads_dir: Path) -> list[Path]:
    """Return source documents only; extracted image assets have no text to index."""
    return sorted(
        path for path in uploads_dir.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_DOCUMENT_SUFFIXES
    )


def build_text_records(source_path: Path) -> tuple[list[dict], list[str]]:
    """Reuse parsing and chunking from the established ingestion pipeline."""
    asset = parse_document(source_path, document_id=None, storage_dir=source_path.parent)
    chunks = chunk_document_pages(
        document_id=asset.document_id,
        filename=asset.filename,
        source_path=asset.source_path,
        pages=[(chunk.page, chunk.content) for chunk in asset.text_chunks],
    )
    records = [
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
            },
        }
        for chunk in chunks
    ]
    return records, [chunk.content for chunk in chunks]


def reindex_text(uploads_dir: Path, chroma_dir: Path, collection_name: str) -> dict[str, int | str]:
    if collection_name == "text_chunks":
        raise ValueError("Refusing to write to text_chunks. Use a new collection such as text_chunks_v2.")
    if not uploads_dir.is_dir():
        raise FileNotFoundError(f"Uploads directory does not exist: {uploads_dir}")

    settings = get_settings()
    vectorstore = ChromaVectorStore(
        persist_directory=chroma_dir,
        text_collection_name=collection_name,
        include_images=False,
    )
    provider = build_default_text_embedding_provider(settings)

    documents_processed = 0
    chunks_created = 0
    embeddings_generated = 0
    for source_path in iter_text_documents(uploads_dir):
        records, texts = build_text_records(source_path)
        documents_processed += 1
        if not records:
            continue
        embeddings = provider.embed_texts(texts)
        vectorstore.upsert_text_chunks(records, embeddings)
        chunks_created += len(records)
        embeddings_generated += len(embeddings)

    return {
        "documents_processed": documents_processed,
        "chunks_created": chunks_created,
        "embeddings_generated": embeddings_generated,
        "embedding_dimension": provider.get_embedding_dimension(),
        "collection_name": collection_name,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Safely reindex text into a new Chroma collection.")
    parser.add_argument("--uploads-dir", type=Path, default=settings.upload_dir)
    parser.add_argument("--chroma-dir", type=Path, default=settings.chroma_db_path)
    parser.add_argument("--collection", default="text_chunks_v2")
    args = parser.parse_args()

    summary = reindex_text(args.uploads_dir, args.chroma_dir, args.collection)
    print(f"documents processed: {summary['documents_processed']}")
    print(f"chunks created: {summary['chunks_created']}")
    print(f"embeddings generated: {summary['embeddings_generated']}")
    print(f"embedding dimension: {summary['embedding_dimension']}")
    print(f"collection name: {summary['collection_name']}")


if __name__ == "__main__":
    main()
