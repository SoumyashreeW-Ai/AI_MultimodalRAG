# Embedding Instructions

## Text

Use a configurable text embedding provider.

The application must not depend on a single hardcoded model.

Support batch embedding.

Load models once and reuse them.

Do not load a model for every query.

---

# Images

Use a CLIP-compatible multimodal embedding model or equivalent provider.

The provider must support:

text → embedding

image → embedding

This enables text-to-image retrieval.

---

# Important

Do not assume text embeddings and image embeddings from unrelated models are comparable.

Text retrieval and image retrieval should initially be performed independently.

Then normalize/fuse their retrieval results.

---

# Interfaces

Implement provider abstractions such as:

TextEmbeddingProvider

ImageEmbeddingProvider

MultimodalEmbeddingProvider

---

# Testing

Normal tests must mock embedding providers.

Do not download large models during unit tests.

Model loading should be separately testable.