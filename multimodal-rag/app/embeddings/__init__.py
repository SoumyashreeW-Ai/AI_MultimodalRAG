"""Embedding provider abstractions and default implementations."""

from .providers import (
    DefaultImageEmbeddingProvider,
    DefaultMultimodalEmbeddingProvider,
    DefaultTextEmbeddingProvider,
    ImageEmbeddingProvider,
    MultimodalEmbeddingProvider,
    TextEmbeddingProvider,
)

__all__ = [
    "TextEmbeddingProvider",
    "ImageEmbeddingProvider",
    "MultimodalEmbeddingProvider",
    "DefaultTextEmbeddingProvider",
    "DefaultImageEmbeddingProvider",
    "DefaultMultimodalEmbeddingProvider",
]
