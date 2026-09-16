from __future__ import annotations

from typing import Any, Sequence

from app.embeddings.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from app.models.retrieval_models import RetrievalResult
from time import perf_counter
from app.observability.opik import get_opik
import json


class RetrievalService:
    """Execute separate text and image retrieval, score, and fuse results."""

    def __init__(
        self,
        vectorstore: Any,
        text_embedding_provider: TextEmbeddingProvider,
        multimodal_embedding_provider: ImageEmbeddingProvider | None = None,
        image_embedding_provider: ImageEmbeddingProvider | None = None,
        text_weight: float = 1.0,
        image_weight: float = 1.0,
        top_k: int = 5,
        candidate_k: int = 20,
        rerank: bool = False,
    ):
        self.vectorstore = vectorstore
        self.text_embedding_provider = text_embedding_provider
        if multimodal_embedding_provider is not None:
            self.multimodal_embedding_provider = multimodal_embedding_provider
        else:
            self.multimodal_embedding_provider = image_embedding_provider
        self.image_embedding_provider = self.multimodal_embedding_provider
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rerank = rerank

    def _normalize_scores(self, results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chroma cosine distances to descending-friendly similarity scores.

        The configured Chroma collections use cosine distance, where lower values
        are more similar.  Keep the original ``score`` untouched for diagnostics
        and place the derived similarity in the existing internal score field.
        """
        if not results:
            return []

        for item in results:
            cosine_distance = float(item.get("score", 0.0))
            similarity_score = 1.0 - cosine_distance
            item["similarity_score"] = similarity_score
            # Preserve this key for the existing fusion path.
            item["normalized_score"] = similarity_score
        return list(results)

    def _deduplicate(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        unique: dict[str, RetrievalResult] = {}
        for result in results:
            key = result.id
            if key not in unique or result.score > unique[key].score:
                unique[key] = result
        return sorted(unique.values(), key=lambda item: item.score, reverse=True)

    def _rerank(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        if not self.rerank:
            return list(results)
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _convert_text_hit(self, hit: dict[str, Any]) -> RetrievalResult:
        metadata = hit.get("metadata") or {}
        return RetrievalResult(
            id=str(hit.get("id") or metadata.get("chunk_id") or "text-unknown"),
            score=float(hit.get("score", 0.0)),
            content_type="text",
            document_id=hit.get("document_id") or metadata.get("document_id"),
            filename=hit.get("filename") or metadata.get("filename"),
            page=hit.get("page") or metadata.get("page"),
            chunk_id=hit.get("chunk_id") or metadata.get("chunk_id"),
            text=hit.get("text") or metadata.get("text"),
            metadata=metadata,
        )

    def _convert_image_hit(self, hit: dict[str, Any]) -> RetrievalResult:
        metadata = hit.get("metadata") or {}
        return RetrievalResult(
            id=str(hit.get("id") or metadata.get("image_id") or "image-unknown"),
            score=float(hit.get("score", 0.0)),
            content_type="image",
            document_id=hit.get("document_id") or metadata.get("document_id"),
            filename=hit.get("filename") or metadata.get("filename"),
            page=hit.get("page") or metadata.get("page"),
            chunk_id=metadata.get("chunk_id"),
            text=metadata.get("caption"),
            image_path=metadata.get("image_path") or hit.get("image_path"),
            metadata=metadata,
        )

    def query(self, query: str, top_k: int | None = None, debug: bool = False) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_top_k = top_k or self.top_k
        candidate_k = max(query_top_k, self.candidate_k)

        text_query_vector = self.text_embedding_provider.embed_text(query)
        image_query_vector = self.multimodal_embedding_provider.embed_text(query)

        opik = get_opik()
        t0 = perf_counter()
        text_hits = self.vectorstore.query_text(query_vector=text_query_vector, n_results=candidate_k)
        t_text = perf_counter() - t0
        t0_img = perf_counter()
        image_hits = self.vectorstore.query_images(query_vector=image_query_vector, n_results=candidate_k)
        t_image = perf_counter() - t0_img
        if opik:
            opik.log_metric("retrieval.text_time_sec", t_text)
            opik.log_metric("retrieval.image_time_sec", t_image)
            opik.log_metric("retrieval.text_hits", len(text_hits or []))
            opik.log_metric("retrieval.image_hits", len(image_hits or []))
            if text_hits:
                top = text_hits[0]
                opik.log_text("retrieval.top_hit_meta", json.dumps({
                    "id": top.get("id"),
                    "filename": (top.get("metadata") or {}).get("filename"),
                }))
            # Log top-k ids and filenames for the query (safe metadata only)
            try:
                topk = [h.get("id") or ((h.get("metadata") or {}).get("chunk_id")) for h in (text_hits or [])][:query_top_k]
                topk_files = [((h.get("metadata") or {}).get("filename")) for h in (text_hits or [])][:query_top_k]
                opik.log_text("retrieval.top_k_ids", json.dumps(topk))
                opik.log_text("retrieval.top_k_filenames", json.dumps(topk_files))
            except Exception:
                pass

        if not text_hits and not image_hits:
            # Fallback: if semantic search yields no hits, attempt a metadata-only
            # listing of available text chunks so the caller can inspect sources.
            try:
                raw = []
                collection = getattr(self.vectorstore, "collections", {}).get("text_chunks")
                if collection is not None:
                    data = collection.get(include=["documents", "metadatas"]) or {}
                    docs = data.get("documents", []) or []
                    mets = data.get("metadatas", []) or []
                    for idx, doc in enumerate(docs):
                        meta = mets[idx] if idx < len(mets) else {}
                        raw.append(
                            {
                                "id": meta.get("chunk_id") or meta.get("document_id") or f"fallback-{idx}",
                                "document_id": meta.get("document_id"),
                                "filename": meta.get("filename"),
                                "page": meta.get("page"),
                                "chunk_id": meta.get("chunk_id"),
                                "content_type": meta.get("content_type"),
                                "text": doc,
                                "score": 0.0,
                                "metadata": meta,
                            }
                        )
            except Exception:
                raw = []

            return {"results": [], "debug": {"text_hits": raw, "image_hits": []} if debug else None}

        normalized_text = self._normalize_scores(text_hits)
        normalized_images = self._normalize_scores(image_hits)

        combined: list[RetrievalResult] = []
        for hit in normalized_text:
            result = self._convert_text_hit(hit)
            result.score = hit.get("normalized_score", 0.0) * self.text_weight
            combined.append(result)

        for hit in normalized_images:
            result = self._convert_image_hit(hit)
            result.score = hit.get("normalized_score", 0.0) * self.image_weight
            combined.append(result)

        deduped = self._deduplicate(combined)
        reranked = self._rerank(deduped)
        final_results = reranked[:query_top_k]

        payload = {
            "results": [
                {
                    "id": item.id,
                    "score": item.score,
                    "content_type": item.content_type,
                    "document_id": item.document_id,
                    "filename": item.filename,
                    "page": item.page,
                    "chunk_id": item.chunk_id,
                    "text": item.text,
                    "image_path": item.image_path,
                    "metadata": item.metadata,
                }
                for item in final_results
            ]
        }

        if debug:
            payload["debug"] = {
                "text_hits": normalized_text,
                "image_hits": normalized_images,
            }

        return payload
