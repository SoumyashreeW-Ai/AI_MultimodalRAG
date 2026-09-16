"""ChromaDB client creation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("chromadb is required for vectorstore functionality") from exc


def create_chroma_client(persist_directory: str | Path = "data/chroma") -> Any:
    """Create a persistent ChromaDB client.

    Uses the modern PersistentClient when available, with a compatibility fallback for
    older ChromaDB versions.
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    try:
        return chromadb.PersistentClient(path=str(persist_path))
    except AttributeError:
        settings = chromadb.config.Settings(
            persist_directory=str(persist_path),
            anonymized_telemetry=False,
        )
        return chromadb.Client(settings)
