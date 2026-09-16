# Architecture Rules

## Required Layers

Use the following logical layers:

app/
├── api/
├── core/
├── models/
├── ingestion/
├── embeddings/
├── vectorstore/
├── retrieval/
├── generation/
└── utils/

frontend/
├── streamlit_app.py
├── api_client.py
└── components/

---

# Dependency Direction

Preferred dependency direction:

API
→ Services
→ Domain logic
→ Infrastructure

Infrastructure includes:

- ChromaDB
- embedding models
- LLM providers
- filesystem

API routes must not directly manipulate ChromaDB.

API routes must not directly load embedding models.

API routes must not directly call the LLM.

---

# Provider Abstraction

External AI services must be accessed through interfaces/providers.

Examples:

TextEmbeddingProvider

ImageEmbeddingProvider

ImageCaptionProvider

LLMProvider

VisionLLMProvider

This allows models/providers to be replaced without rewriting the RAG pipeline.

---

# Domain Models

Use explicit models for:

Document

TextChunk

ImageAsset

RetrievalResult

SourceReference

RAGAnswer

Avoid passing arbitrary dictionaries throughout the application when a typed model is appropriate.

---

# Persistence

ChromaDB stores embeddings and metadata.

The original uploaded files remain in the filesystem or future object storage.

ChromaDB should not be treated as the source of truth for binary files.

---

# No Circular Dependencies

Avoid circular imports.

Infrastructure must not depend on API routes.

Domain models should remain lightweight.