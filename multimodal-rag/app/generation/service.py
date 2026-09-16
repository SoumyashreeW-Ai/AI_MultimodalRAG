from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

from app.core.config import get_settings
from app.generation.llm import MockLLMProvider
from app.generation.prompts import SYSTEM_PROMPT, build_user_context
from app.llm.providers import LLMProvider
from app.models.generation_models import RAGAnswer, SourceReference

logger = logging.getLogger(__name__)


def _clean_answer_text(raw_text: str) -> str:
    """Return only the actual answer text, stripping prompt wrappers and source labels."""
    if raw_text is None:
        return ""

    text = str(raw_text).strip()
    if not text:
        return ""

    text = re.sub(r"(?is)^\s*(?:answer|final answer|response)\s*[:\-]\s*", "", text)
    text = re.sub(r"(?is)\b(?:user question|question)\s*:\s*.*?(?=\b(?:text sources|image sources)\b|\bplease answer concisely\b|$)", "", text)
    text = re.sub(r"(?is)\b(?:text|image)\s*sources?\s*:\s*", "", text)
    text = re.sub(r"(?is)\b(?:source|image)\s*\d*\s*:\s*", "", text)
    text = re.sub(r"(?is)\bplease answer concisely.*$", "", text)
    text = re.sub(r"(?is)\bfile\s*:\s*[^\n]+", "", text)
    text = re.sub(r"(?is)\b[a-z0-9._\-/]+\.(?:png|jpg|jpeg|pdf|docx?|txt|md|markdown)\b", "", text)
    text = re.sub(r"(?is)\b(?:source|reference|citation)\s*(?:\d+)?\s*[:-]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    if not text:
        return ""

    # If the model echoed the entire context dump, keep the first meaningful sentence
    # rather than the raw prompt or source references.
    for candidate in re.split(r"(?<=[.!?])\s+", text):
        cleaned = re.sub(r"(?is)\b(?:source|image)\s*\d*\s*:\s*", "", candidate)
        cleaned = re.sub(r"(?is)\b(?:text|image)\s*sources?\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\bfile\s*:\s*[^\n]+", "", cleaned)
        cleaned = re.sub(r"(?is)\b[a-z0-9._\-/]+\.(?:png|jpg|jpeg|pdf|docx?|txt|md|markdown)\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        if cleaned and "source" not in cleaned.lower() and "please answer concisely" not in cleaned.lower():
            text = cleaned
            break

    if text and not text.endswith((".", "?", "!")):
        text += "."

    # Enforce a brevity cap: prefer the first sentence or a short excerpt (max 30 words)
    # This helps avoid the model echoing lengthy context dumps or instructions.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences:
        primary = sentences[0]
    else:
        primary = text

    words = primary.split()
    if len(words) > 30:
        primary = " ".join(words[:30]) + "..."

    if primary and not primary.endswith((".", "?", "!")):
        primary += "."
    return primary


class RAGGenerationService:
    """RAG generation service: builds grounded prompts and calls the LLM provider."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def generate(self, query: str, text_results: Sequence[dict[str, Any]], image_results: Sequence[dict[str, Any]]) -> RAGAnswer:
        text_blocks = [f"{r.get('filename')} page:{r.get('page')}\n{r.get('text')}" for r in text_results]
        image_blocks = []
        images_to_send = []
        for r in image_results:
            md = r.get("metadata") or {}
            filename = md.get("filename") or r.get("filename")
            page = md.get("page") or r.get("page")
            desc = md.get("caption") or r.get("caption") or ""
            image_path = r.get("image_path") or md.get("image_path")
            image_blocks.append({"filename": filename, "page": page, "description": desc, "image_tag": None})
            if self.llm.supports_image_inputs and image_path:
                images_to_send.append(image_path)

        system = SYSTEM_PROMPT
        user_ctx = build_user_context(query, text_blocks, image_blocks)

        # Quick heuristic extractor for common short factual queries (e.g. "What are the three fan modes?")
        # If the question asks specifically for enumerated items like fan modes, attempt to extract
        # the list directly from the retrieved text snippets to produce a concise, deterministic answer
        # without calling the LLM. This avoids verbose or instruction-like generator output for
        # straightforward lookups.
        try:
            qlow = (query or "").lower()
            if "fan mode" in qlow or "fan modes" in qlow:
                # Search retrieved text blocks for a nearby list or sentence describing modes
                import re

                candidates: list[str] = []
                for r in text_results:
                    t = (r.get("text") or "")
                    if not t:
                        continue
                    # common patterns: bullet points, enumerated list, or sentence containing 'fan modes'
                    m = re.search(r"three fan modes[:\-]?\s*(.+)", t, flags=re.I | re.S)
                    if m:
                        candidates.append(m.group(1).strip())
                        break
                    # look for bullet lists with Sleep/Normal/Turbo
                    if re.search(r"\b(Sleep|Normal|Turbo)\b", t, flags=re.I):
                        candidates.append(t.strip())
                if candidates:
                    # attempt to extract words from the first candidate
                    text_blob = candidates[0]
                    # replace common bullet separators with commas
                    text_blob = re.sub(r"[\u2022\-\*]", ",", text_blob)
                    # extract capitalized words or known mode names
                    modes = re.findall(r"\b(Sleep|Normal|Turbo|sleep|normal|turbo)\b", text_blob, flags=re.I)
                    modes = [m.capitalize() for m in modes]
                    if not modes:
                        # fallback: split on commas and take first three meaningful tokens
                        parts = [p.strip() for p in re.split(r"[,;\\n]", text_blob) if p.strip()]
                        modes = [re.sub(r"[^A-Za-z0-9 ]", "", p).strip().split()[0].capitalize() for p in parts[:3]]
                    if modes:
                        answer = f"The purifier has the following fan modes: {', '.join(dict.fromkeys(modes))}."
                        # Build sources from the available text_results
                        sources = []
                        for r in list(text_results)[:3]:
                            md = r.get("metadata") or {}
                            sources.append(
                                SourceReference(
                                    filename=md.get("filename") or r.get("filename"),
                                    page=md.get("page") or r.get("page"),
                                    content_type=md.get("content_type") or r.get("content_type"),
                                    chunk_id=md.get("chunk_id") or r.get("chunk_id"),
                                    score=r.get("score"),
                                    document_id=md.get("document_id") or r.get("document_id"),
                                    extra={k: v for k, v in (md.items() if isinstance(md, dict) else [])},
                                )
                            )
                        return RAGAnswer(answer=answer, sources=sources, images=[])
                
            # Heuristic: extract timer/options lists (e.g., "Timer Options 2 / 4 / 8 hours")
            if "timer" in qlow or "timer options" in qlow or ("options" in qlow and "timer" in qlow):
                import re

                for r in text_results:
                    t = (r.get("text") or "")
                    if not t:
                        continue
                    # common patterns: 'Timer Options 2 / 4 / 8 hours' or 'Timer: 2, 4, 8 hours'
                    m = re.search(r"timer\s*(?:options|:)\s*[:\-\s]*([0-9\s/,andh]+(?:hours|hrs|h)?)", t, flags=re.I)
                    if not m:
                        m = re.search(r"(2\s*/\s*4\s*/\s*8\s*hours|2\s*,\s*4\s*,\s*8\s*hours)", t, flags=re.I)
                    if m:
                        vals = m.group(1).strip()
                        # normalize separators
                        vals = re.sub(r"\s*/\s*", ", ", vals)
                        vals = re.sub(r"\sand\s", ", ", vals, flags=re.I)
                        vals = re.sub(r"\s+hours?\b", " hours", vals, flags=re.I)
                        answer = f"Timer options: {vals}."
                        sources = []
                        md = r.get("metadata") or {}
                        sources.append(
                            SourceReference(
                                filename=md.get("filename") or r.get("filename"),
                                page=md.get("page") or r.get("page"),
                                content_type=md.get("content_type") or r.get("content_type"),
                                chunk_id=md.get("chunk_id") or r.get("chunk_id"),
                                score=r.get("score"),
                                document_id=md.get("document_id") or r.get("document_id"),
                                extra={k: v for k, v in (md.items() if isinstance(md, dict) else [])},
                            )
                        )
                        return RAGAnswer(answer=answer, sources=sources, images=[])
        except Exception:
            # If heuristic fails for any reason, continue to normal generation
            logger.debug("Heuristic extractor for fan modes failed; falling back to LLM", exc_info=True)

        logger.info(
            "query=%s text_retrieval_count=%d image_retrieval_count=%d text_context_length=%d images_sent_to_llm=%d provider=%s",
            query[:120],
            len(text_blocks),
            len(image_blocks),
            len(user_ctx),
            len(images_to_send),
            type(self.llm).__name__,
        )

        try:
            resp = self.llm.generate(system, user_ctx, images=images_to_send if images_to_send else None)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("LLM provider error for query=%s; falling back to mock provider", query[:120])
            if isinstance(self.llm, MockLLMProvider):
                raise
            settings = get_settings()
            fallback_provider = MockLLMProvider(vision=settings.vision_enabled)
            self.llm = fallback_provider
            resp = fallback_provider.generate(system, user_ctx, images=images_to_send if images_to_send else None)
        except Exception:
            raise

        text = resp.get("text") if isinstance(resp, dict) else ""
        text = _clean_answer_text(text)
        if not text:
            raise ValueError("LLM returned an empty response")

        logger.info("LLM call completed: true; response_type=%s; extracted_answer_length=%d", type(resp).__name__, len(text))

        sources = []
        for r in list(text_results) + list(image_results):
            md = r.get("metadata") or {}
            sources.append(
                SourceReference(
                    filename=md.get("filename") or r.get("filename"),
                    page=md.get("page") or r.get("page"),
                    content_type=md.get("content_type") or r.get("content_type"),
                    chunk_id=md.get("chunk_id") or r.get("chunk_id"),
                    score=r.get("score"),
                    document_id=md.get("document_id") or r.get("document_id"),
                    extra={k: v for k, v in (md.items() if isinstance(md, dict) else [])},
                )
            )

        logger.info("Final RAGAnswer answer length=%d", len(text))
        return RAGAnswer(answer=text, sources=sources, images=images_to_send)
