from unittest.mock import Mock

import pytest

from app.embeddings.providers import (
    DefaultImageEmbeddingProvider,
    DefaultMultimodalEmbeddingProvider,
    DefaultTextEmbeddingProvider,
)


def test_default_text_provider_loads_once_and_batches():
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [[float(len(text)), 1.0] for text in texts]

    provider = DefaultTextEmbeddingProvider(model_loader=lambda: model)
    vectors = provider.embed_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 2
    assert vectors[0][1] == 1.0
    assert vectors[0][0] > 0
    assert provider.model is model
    assert provider.model_load_count == 1


def test_text_embeddings_are_semantic_shape_stable_and_cached():
    class FakeSentenceTransformer:
        def __init__(self):
            self.load_calls = 0

        def encode(self, texts, **kwargs):
            vectors = {
                "What is deep learning?": [0.1, 0.2, 0.9],
                "What is generative AI?": [0.8, 0.1, 0.3],
            }
            return [vectors[text] for text in texts]

        def get_sentence_embedding_dimension(self):
            return 3

    model = FakeSentenceTransformer()
    provider = DefaultTextEmbeddingProvider(model_loader=lambda: model)

    deep_learning = provider.embed_text("What is deep learning?")
    generative_ai = provider.embed_text("What is generative AI?")
    deep_learning_again = provider.embed_text("What is deep learning?")

    assert provider.get_embedding_dimension() > 2
    assert deep_learning != generative_ai
    assert deep_learning == deep_learning_again
    assert provider.model_load_count == 1


def test_text_provider_fails_loudly_without_sentence_transformer_interface():
    provider = DefaultTextEmbeddingProvider(model_loader=lambda: object())

    with pytest.raises(RuntimeError, match="SentenceTransformer.encode"):
        provider.embed_text("What is deep learning?")


def test_multimodal_provider_rejects_configuration_dictionary_before_encode():
    provider = DefaultMultimodalEmbeddingProvider(
        model_loader=lambda: {"model_name": "clip", "kind": "multimodal"}
    )

    with pytest.raises(RuntimeError, match="configuration dictionary"):
        provider.embed_text("What is deep learning?")


def test_default_image_provider_batches_and_reuses_model():
    model = Mock()
    model.encode_image.side_effect = lambda paths, **kwargs: [[float(len(path)), 2.0] for path in paths]

    provider = DefaultImageEmbeddingProvider(model_loader=lambda: model)
    vectors = provider.embed_images(["/tmp/a.png", "/tmp/b.png"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 2
    assert vectors[0][1] == 2.0
    assert vectors[0][0] > 0
    assert provider.model is model
    assert provider.model_load_count == 1


def test_multimodal_provider_embeds_text_and_images():
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [[float(len(text)), 3.0] for text in texts]
    model.encode_image.side_effect = lambda paths, **kwargs: [[float(len(path)), 4.0] for path in paths]

    provider = DefaultMultimodalEmbeddingProvider(model_loader=lambda: model)
    text_vectors = provider.embed_texts(["hello", "world"])
    image_vectors = provider.embed_images(["p1.png", "p2.png"])

    assert len(text_vectors) == 2
    assert len(image_vectors) == 2
    assert all(vector[1] == 3.0 for vector in text_vectors)
    assert all(vector[1] == 4.0 for vector in image_vectors)
    assert all(vector[0] > 0 for vector in text_vectors + image_vectors)
    assert provider.model is model
