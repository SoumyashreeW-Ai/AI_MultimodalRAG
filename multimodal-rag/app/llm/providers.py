from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class LLMProvider(ABC):
    """Abstract interface for LLM providers used by the RAG generation service."""

    @property
    @abstractmethod
    def supports_image_inputs(self) -> bool:
        """Whether the provider accepts image inputs natively."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load and cache the model."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, images: Sequence[str] | None = None) -> dict[str, Any]:
        """Generate a response given prompts and optional images.

        Returns a dictionary with keys: `text` (str) and optionally `debug`.
        """


class DefaultLLMProvider(LLMProvider):
    """A minimal provider used for testing/fallback.

    The `model_loader` should return either a callable that accepts `(system, prompt, images)`
    and returns a string, or an object with a `generate` method.
    """

    def __init__(self, model_loader=None):
        self.model_loader = model_loader
        self._model = None

    @property
    def supports_image_inputs(self) -> bool:
        # Default provider does not support images
        return False

    def load_model(self) -> Any:
        if self._model is None and self.model_loader is not None:
            self._model = self.model_loader()
        return self._model

    def generate(self, system_prompt: str, user_prompt: str, images: Sequence[str] | None = None) -> dict[str, Any]:
        model = self.load_model()
        if model is None:
            # deterministic fallback
            return {"text": user_prompt}
        if hasattr(model, "generate"):
            return {"text": model.generate(system_prompt, user_prompt, images)}
        if callable(model):
            return {"text": model(system_prompt, user_prompt, images)}
        return {"text": user_prompt}
