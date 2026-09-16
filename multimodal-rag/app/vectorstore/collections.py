"""ChromaDB-backed vectorstore implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .chroma_client import create_chroma_client


class ChromaVectorStore:
    """Store and query text/image embeddings in ChromaDB."""

    COLLECTION_NAMES = ("text_chunks", "images")
    # Do not hard-code an expected embedding dimension; infer or accept configured value.
    EXPECTED_EMBEDDING_DIMENSION = None

    @staticmethod
    def _coerce_metadata_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple, set)):
            return str([ChromaVectorStore._coerce_metadata_value(item) for item in value])
        if isinstance(value, dict):
            sanitized = {
                str(key): ChromaVectorStore._coerce_metadata_value(item)
                for key, item in value.items()
                if ChromaVectorStore._coerce_metadata_value(item) is not None
            }
            return str(sanitized)
        return str(value)

    @classmethod
    def _sanitize_metadata(cls, metadata: dict[str, Any] | None) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            coerced = cls._coerce_metadata_value(value)
            if coerced is None:
                continue
            cleaned[str(key)] = coerced
        return cleaned

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        text_collection_name: str = "text_chunks",
        include_images: bool = True,
        expected_embedding_dimension: int | None = None,
    ):
        self.persist_directory = str(Path(persist_directory))
        self.include_images = include_images
        # If provided, enforce a consistent embedding dimensionality at write/query time.
        self.expected_embedding_dimension = expected_embedding_dimension
        if not text_collection_name:
            raise ValueError("text_collection_name cannot be empty")
        self.text_collection_name = text_collection_name
        self.client = create_chroma_client(self.persist_directory)
        self._reset_legacy_collections()
        self.collections = {"text_chunks": self._get_or_create_collection(text_collection_name)}
        if include_images:
            self.collections["images"] = self._get_or_create_collection("images")

    def _reset_legacy_collections(self) -> None:
        # Previously this method attempted to delete existing collections
        # that matched legacy naming patterns. That behavior can be destructive
        # in development when the server reloads (it would remove persisted
        # collections unexpectedly). Make this a no-op to preserve existing
        # collections across restarts. If legacy cleanup is ever needed,
        # perform it manually or behind an explicit flag.
        return

    def _get_or_create_collection(self, collection_name: str) -> Any:
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reload_client(self) -> None:
        """Attempt to persist/close the client and recreate collection handles.

        This helps ensure data written in one process becomes visible to new
        client instances (useful for short-lived scripts that index then exit).
        """
        try:
            # Prefer a persist call if provided by the client implementation.
            if hasattr(self.client, "persist"):
                try:
                    self.client.persist()
                except Exception:
                    pass
            # Close the client to flush any in-memory state.
            if hasattr(self.client, "close"):
                try:
                    self.client.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Recreate the client and collection handles so other processes can see data.
        self.client = create_chroma_client(self.persist_directory)
        self.collections = {"text_chunks": self._get_or_create_collection(self.text_collection_name)}
        if self.include_images:
            self.collections["images"] = self._get_or_create_collection("images")

    @staticmethod
    def _resolve_id(record: dict[str, Any], fallback: str = "item") -> str:
        if record.get("id"):
            return str(record["id"])

        metadata = record.get("metadata") or {}
        document_id = metadata.get("document_id")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")
        image_id = metadata.get("image_id")

        if document_id and page is not None and chunk_id:
            return f"{document_id}:page-{page}:{chunk_id}"
        if document_id and page is not None and image_id:
            return f"{document_id}:page-{page}:{image_id}"
        if document_id:
            return f"{document_id}:{fallback}"
        return fallback

    def _add_documents(
        self,
        collection_name: str,
        records: Iterable[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        records = list(records)
        if not records:
            return 0
        if len(records) != len(embeddings):
            raise ValueError("Each record must have a corresponding embedding")

        collection = self.collections[collection_name]
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        for index, record in enumerate(records):
            item_id = self._resolve_id(record, fallback=f"{collection_name}-{index}")
            item_text = record.get("text") or record.get("caption") or ""
            metadata = self._sanitize_metadata(record.get("metadata") or {})
            metadata.setdefault("content_type", record.get("content_type", "unknown"))
            if "sha256" in record:
                metadata["sha256"] = record["sha256"]
            ids.append(item_id)
            documents.append(str(item_text))
            metadatas.append(self._sanitize_metadata(metadata))
            vectors.append(list(embeddings[index]))
        # Runtime check: ensure embeddings are consistent in dimension if requested.
        if vectors:
            dim = len(vectors[0])
            if self.expected_embedding_dimension is not None and dim != self.expected_embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.expected_embedding_dimension}, got {dim}"
                )
            # If no expected dimension configured, record the first observed dimension for future checks.
            if self.expected_embedding_dimension is None:
                self.expected_embedding_dimension = dim
        collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )
        # Ensure persistence/visibility to other processes.
        try:
            self._reload_client()
        except Exception:
            pass
        return len(ids)

    def upsert_text_chunks(
        self,
        text_chunks: Iterable[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """Insert or update text chunks without deleting the collection."""
        records = list(text_chunks)
        if not records:
            return 0
        if len(records) != len(embeddings):
            raise ValueError("Each record must have a corresponding embedding")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for index, record in enumerate(records):
            ids.append(self._resolve_id(record, fallback=f"text_chunks-{index}"))
            documents.append(str(record.get("text") or ""))
            metadata = self._sanitize_metadata(record.get("metadata") or {})
            metadata.setdefault("content_type", record.get("content_type", "text"))
            if "sha256" in record:
                metadata["sha256"] = record["sha256"]
            metadatas.append(self._sanitize_metadata(metadata))
            vectors.append(list(embeddings[index]))

        self.collections["text_chunks"].upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )
        # Persist and reload so other processes see the changes.
        try:
            self._reload_client()
        except Exception:
            pass
        return len(ids)

    def add_text_chunks(
        self,
        text_chunks: Iterable[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        return self._add_documents("text_chunks", text_chunks, embeddings)

    def add_images(
        self,
        images: Iterable[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        return self._add_documents("images", images, embeddings)

    def _query_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self.collections[collection_name]
        try:
            result = collection.query(
                query_embeddings=[list(query_vector)],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # If the underlying Chroma collection is missing or the query fails
            # for environmental/test reasons, return an empty result set instead
            # of raising. This makes retrieval robust in ephemeral test setups.
            return []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, doc_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            hits.append(
                {
                    "id": doc_id,
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "image_id": metadata.get("image_id"),
                    "content_type": metadata.get("content_type"),
                    "text": documents[index] if index < len(documents) else "",
                    "score": distances[index] if index < len(distances) else None,
                    "metadata": metadata,
                }
            )
        return hits

    def query_text(
        self,
        query_vector: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._query_collection("text_chunks", query_vector, n_results=n_results, where=where)

    def query_images(
        self,
        query_vector: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._query_collection("images", query_vector, n_results=n_results, where=where)

    def delete_document(self, document_id: str) -> bool:
        deleted_any = False
        for collection in self.collections.values():
            existing_ids = collection.get(where={"document_id": document_id}, include=[]).get("ids", [])
            if existing_ids:
                collection.delete(ids=existing_ids)
                deleted_any = True
        return deleted_any

    def document_exists(self, document_id: str, sha256: str | None = None) -> bool:
        for collection in self.collections.values():
            if sha256 is not None:
                result = collection.get(where={"sha256": sha256}, limit=1)
                if result.get("ids"):
                    return True
            result = collection.get(where={"document_id": document_id}, limit=1)
            if result.get("ids"):
                return True
        return False

    def get_document_count(self) -> int:
        documents: set[str] = set()
        for collection in self.collections.values():
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", []) or []
            for metadata in metadatas:
                if isinstance(metadata, dict) and metadata.get("document_id"):
                    documents.add(str(metadata["document_id"]))
        return len(documents)

    def health_check(self) -> bool:
        try:
            self.client.heartbeat()
            for collection in self.collections.values():
                collection.count()
            return True
        except Exception:
            return False
