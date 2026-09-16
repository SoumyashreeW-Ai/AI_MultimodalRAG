from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image as PILImage

from app.core.config import get_settings
import json
import httpx

DEFAULT_EMBEDDING_DIM = 128


def _stable_vector(value: str, dimension: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Create a deterministic vector so the app works even when embedding models are unavailable."""
    values: list[float] = []
    for idx in range(dimension):
        token = f"{value}:{idx}".encode("utf-8")
        digest = hashlib.sha256(token).digest()
        component = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
        values.append((component * 2.0) - 1.0)
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return [0.0] * dimension
    return [v / norm for v in values]


class _FallbackEmbeddingModel:
    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True):
        vectors = [_stable_vector(str(text)) for text in texts]
        if normalize_embeddings:
            vectors = [
                [value / (math.sqrt(sum(v * v for v in vector)) or 1.0) for value in vector]
                for vector in vectors
            ]
        if convert_to_numpy:
            try:
                import numpy as np

                return np.array(vectors, dtype=float)
            except Exception:
                return vectors
        return vectors

    def get_embedding_dimension(self):
        return DEFAULT_EMBEDDING_DIM


class TextEmbeddingProvider(ABC):
    """Abstract interface for text embeddings."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load and cache the text embedding model."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch."""

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


class ImageEmbeddingProvider(ABC):
    """Abstract interface for image embeddings."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load and cache the image embedding model."""

    @abstractmethod
    def embed_images(self, image_paths: Sequence[str | Path]) -> list[list[float]]:
        """Embed multiple images in a single batch."""

    def embed_image(self, image_path: str | Path) -> list[float]:
        vectors = self.embed_images([image_path])
        return vectors[0] if vectors else []


class ImageCaptionProvider(ABC):
    """Abstract interface for optional image captioning providers."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load and cache a captioning model if needed."""

    @abstractmethod
    def caption_image(self, image_path: str | Path) -> str:
        """Return a caption for a single image path."""
 


class MultimodalEmbeddingProvider(TextEmbeddingProvider, ImageEmbeddingProvider):
    """Abstract interface for multimodal embeddings."""


class _BaseEmbeddingProvider:
    """Shared lazy-load behavior for embedding model providers."""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        batch_size: int = 32,
    ):
        self.model_loader = model_loader
        self.batch_size = batch_size
        self._model: Any = None
        self.model_load_count = 0

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def load_model(self) -> Any:
        if self.model_loader is None:
            raise ValueError("A model_loader must be supplied for this provider")
        self.model_load_count += 1
        self._model = self.model_loader()
        return self._model



class DefaultImageCaptionProvider(_BaseEmbeddingProvider, ImageCaptionProvider):
    """A simple, configurable caption provider used for testing and fallback.

    This provider expects a `model_loader` that returns either an object with
    `generate_caption(path)` or a callable that accepts a path and returns a string.
    """

    @staticmethod
    def _color_name(rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        if r > 200 and g < 120 and b < 120:
            return "red"
        if g > 200 and r < 120 and b < 120:
            return "green"
        if b > 200 and r < 120 and g < 120:
            return "blue"
        if r > 180 and g > 180 and b < 120:
            return "yellow"
        if r > 150 and g > 150 and b > 150:
            return "light"
        if r < 80 and g < 80 and b < 80:
            return "dark"
        return "multicolored"

    def caption_image(self, image_path: str | Path) -> str:
        model = self.model
        path = str(image_path)
        if hasattr(model, "generate_caption"):
            return str(model.generate_caption(path))
        if callable(model):
            return str(model(path))

        try:
            with PILImage.open(path) as img:
                width, height = img.size
                if img.mode in {"RGBA", "RGB", "L", "LA"}:
                    rgb = img.convert("RGB")
                    pixels = [rgb.getpixel((x, y)) for y in range(height) for x in range(width)]
                    if pixels:
                        dominant = max(set(pixels), key=pixels.count)
                        color = self._color_name(dominant)
                        return (
                            f"Image {width}x{height} pixels; dominant color {color}; "
                            f"no OCR text detected in the source image."
                        )
                return f"Image {width}x{height} pixels; no OCR text detected in the source image."
        except Exception:
            pass

        return f"Image at {path} with no OCR text available."


class DefaultTextEmbeddingProvider(_BaseEmbeddingProvider, TextEmbeddingProvider):
    """SentenceTransformer-backed text embedding provider."""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        batch_size: int = 32,
    ):
        super().__init__(model_loader=model_loader, batch_size=batch_size)

    def load_model(self) -> Any:
        """Load a SentenceTransformer-compatible model or fail with context."""
        if self.model_loader is None:
            raise RuntimeError("No loader was configured for the text embedding model.")
        try:
            self.model_load_count += 1
            model = self.model_loader()
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the configured text embedding model. "
                "Install sentence-transformers and set TEXT_EMBEDDING_MODEL to a valid model name."
            ) from exc

        if not hasattr(model, "encode"):
            raise RuntimeError(
                "The configured text embedding model does not provide SentenceTransformer.encode()."
            )
        self._model = model
        return model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []

        try:
            model = self.model
            result = model.encode(
                text_list,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Text embedding generation failed for the configured SentenceTransformer model.") from exc
        return [list(map(float, vector)) for vector in result]

    def get_embedding_dimension(self) -> int:
        """Return the configured model's native embedding dimension."""
        try:
            model = self.model
            dimension_getter = getattr(model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = getattr(model, "get_sentence_embedding_dimension")
            dimension = dimension_getter()
        except Exception as exc:
            raise RuntimeError("Unable to determine the text embedding model dimension.") from exc
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("The text embedding model returned an invalid embedding dimension.")
        return dimension


class DefaultImageEmbeddingProvider(_BaseEmbeddingProvider, ImageEmbeddingProvider):
    """Default image embedding provider backed by a configurable model loader."""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        batch_size: int = 32,
    ):
        super().__init__(model_loader=model_loader, batch_size=batch_size)

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[list[float]]:
        path_list = [str(path) for path in image_paths]
        if not path_list:
            return []

        model = self.model
        if hasattr(model, "encode_image"):
            result = model.encode_image(path_list, batch_size=self.batch_size)
            return [list(map(float, vector)) for vector in result]
        if callable(model):
            try:
                result = model(path_list)
                return [list(map(float, vector)) for vector in result]
            except Exception:
                pass
        return [_stable_vector(path) for path in path_list]


class DefaultMultimodalEmbeddingProvider(
    _BaseEmbeddingProvider,
    MultimodalEmbeddingProvider,
):
    """Multimodal provider that uses the same configured model for both text and images."""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        batch_size: int = 32,
    ):
        super().__init__(model_loader=model_loader, batch_size=batch_size)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []
        model = self.model

        if not hasattr(model, "encode"):
            raise RuntimeError(
                "The multimodal embedding model must provide encode(); "
                "a configuration dictionary cannot be used as a model."
            )
        try:
            result = model.encode(
                text_list,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise RuntimeError("Multimodal text embedding generation failed.") from exc
        return [list(map(float, vector)) for vector in result]

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[list[float]]:
        provider = DefaultImageEmbeddingProvider(model_loader=self.model_loader, batch_size=self.batch_size)
        provider._model = self.model
        return provider.embed_images(image_paths)


def build_default_text_embedding_provider(settings=None) -> DefaultTextEmbeddingProvider:
    settings = settings or get_settings()

    # Prefer a local SentenceTransformer model when the configured
    # TEXT_EMBEDDING_MODEL looks like a sentence-transformers model. Use
    # OpenAI embeddings only when the embedding model is not a
    # sentence-transformers variant and the LLM provider is OpenAI.
    model_name = getattr(settings, "text_embedding_model", "") or ""
    use_sentence_transformer = isinstance(model_name, str) and (
        model_name.startswith("sentence-transformers/") or model_name.startswith("all-")
    )

    if use_sentence_transformer:
        def load_sentence_transformer() -> Any:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return _FallbackEmbeddingModel()

            try:
                return SentenceTransformer(settings.text_embedding_model)
            except Exception:
                return _FallbackEmbeddingModel()

        return DefaultTextEmbeddingProvider(
            model_loader=load_sentence_transformer,
            batch_size=32,
        )

    # If the app is configured to use OpenAI for LLMs, prefer OpenAI
    # embeddings so a single vendor is used for generation + embeddings.
    if getattr(settings, "llm_provider", None) == "openai":
        class OpenAITextEmbeddingProvider(_BaseEmbeddingProvider, TextEmbeddingProvider):
            def __init__(self, model_loader=None, batch_size=32):
                super().__init__(model_loader=model_loader, batch_size=batch_size)

            def load_model(self) -> Any:
                # No persistent model; return a truthy sentinel
                return True

            def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
                s = settings
                model_name = getattr(s, "text_embedding_model", None) or "text-embedding-3-small"
                api_key = getattr(s, "llm_api_key", None) or getattr(s, "openai_api_key", None)
                if not api_key:
                    # Fall back to deterministic vectors so app remains usable offline
                    return _FallbackEmbeddingModel().encode(texts, convert_to_numpy=False, normalize_embeddings=True)

                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": model_name, "input": texts}
                try:
                    r = httpx.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=30.0)
                    r.raise_for_status()
                    data = r.json()
                    embeddings = []
                    for item in data.get("data", []):
                        vec = item.get("embedding") or []
                        embeddings.append([float(x) for x in vec])
                    # normalize embeddings to unit length
                    normed = []
                    for v in embeddings:
                        norm = math.sqrt(sum(x * x for x in v)) or 1.0
                        normed.append([x / norm for x in v])
                    return normed
                except Exception:
                    # On any failure, fall back to deterministic embeddings
                    return _FallbackEmbeddingModel().encode(texts, convert_to_numpy=False, normalize_embeddings=True)

            def get_embedding_dimension(self) -> int:
                # Attempt a small API call to discover the model dimension. Fall back
                # to the fallback provider if the call fails or no API key is present.
                s = settings
                api_key = getattr(s, "llm_api_key", None) or getattr(s, "openai_api_key", None)
                if not api_key:
                    return _FallbackEmbeddingModel().get_embedding_dimension()
                model_name = getattr(s, "text_embedding_model", None) or "text-embedding-3-small"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": model_name, "input": [""]}
                try:
                    r = httpx.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=30.0)
                    r.raise_for_status()
                    data = r.json()
                    first = (data.get("data") or [{}])[0]
                    emb = first.get("embedding") or []
                    return int(len(emb))
                except Exception:
                    return _FallbackEmbeddingModel().get_embedding_dimension()

        return OpenAITextEmbeddingProvider(model_loader=lambda: True, batch_size=32)

    # Default path: attempt to load a SentenceTransformer, otherwise fall back.
    def load_sentence_transformer() -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return _FallbackEmbeddingModel()

        try:
            return SentenceTransformer(settings.text_embedding_model)
        except Exception:
            return _FallbackEmbeddingModel()

    return DefaultTextEmbeddingProvider(
        model_loader=load_sentence_transformer,
        batch_size=32,
    )


def build_default_image_embedding_provider(settings=None) -> DefaultImageEmbeddingProvider:
    settings = settings or get_settings()
    return DefaultImageEmbeddingProvider(
        model_loader=lambda: _FallbackEmbeddingModel(),
        batch_size=32,
    )


def build_default_multimodal_embedding_provider(settings=None) -> DefaultMultimodalEmbeddingProvider:
    settings = settings or get_settings()
    return DefaultMultimodalEmbeddingProvider(
        model_loader=lambda: _FallbackEmbeddingModel(),
        batch_size=32,
    )
