# Multimodal RAG - GitHub Copilot Instructions

## Role

You are a senior Python AI engineer building a production-quality multimodal Retrieval Augmented Generation system.

The system must support text and image retrieval and must provide grounded answers using retrieved context.

Do not treat this as a demo-only application. Code must be modular, testable, maintainable, and suitable for local development and future production deployment.

---

# Technology Stack

Use:

- Python 3.11+
- FastAPI
- Pydantic / Pydantic Settings
- Streamlit
- ChromaDB
- pytest
- httpx
- python-dotenv
- PyMuPDF for PDF processing where appropriate
- python-docx for DOCX processing where appropriate
- Pillow for image processing
- Sentence Transformers or configurable embedding provider for text
- CLIP-compatible model for image embeddings
- Configurable LLM provider
- Configurable vision/multimodal LLM provider

Do not tightly couple the application to one LLM or embedding vendor.

---

# Core Architecture

The application must separate:

1. API
2. Ingestion
3. Parsing
4. Chunking
5. Embeddings
6. Vector storage
7. Retrieval
8. Reranking
9. LLM generation
10. Multimodal processing
11. Frontend
12. Configuration
13. Testing

Business logic must not be placed directly inside FastAPI route functions.

Business logic must not be placed directly inside Streamlit UI code.

---

# Main Data Flow

The ingestion pipeline is:

Document
→ File validation
→ Document parsing
→ Text extraction
→ Image extraction
→ Text chunking
→ Text embedding
→ Image embedding
→ Optional image captioning
→ ChromaDB

The query pipeline is:

User query
→ Query embedding
→ Text retrieval
→ Image retrieval
→ Result normalization
→ Result fusion
→ Optional reranking
→ Multimodal context construction
→ LLM
→ Grounded answer
→ Sources and images

---

# Multimodal Requirement

This application must implement genuine multimodal RAG.

Retrieving an image and merely displaying it is NOT sufficient.

Images must participate in the retrieval and reasoning pipeline.

The preferred architecture is:

User query
→ text embedding
→ text retrieval

AND

User query
→ multimodal/CLIP text embedding
→ image retrieval

Then:

Text results + image results
→ context construction
→ multimodal LLM

If the selected LLM cannot process images directly, use image captions/descriptions as a fallback.

---

# Coding Standards

Always:

- Use type hints.
- Use Pydantic models for API contracts.
- Use dependency injection where appropriate.
- Keep functions small and focused.
- Use descriptive names.
- Add docstrings to important public classes/functions.
- Handle errors explicitly.
- Use structured logging.
- Avoid global mutable state.
- Keep configuration centralized.
- Write tests for new functionality.

Do not:

- hardcode API keys
- hardcode filesystem paths
- put secrets in source code
- use wildcard imports
- create unnecessary duplicate classes
- create giant Python files
- put business logic inside Streamlit
- put RAG logic directly inside API routes
- silently swallow exceptions
- claim functionality works without testing

---

# Configuration

All configurable values must come from environment variables or Pydantic Settings.

Examples:

- LLM provider
- API keys
- embedding models
- ChromaDB path
- upload directory
- image directory
- chunk size
- chunk overlap
- top-k values
- reranking configuration
- maximum upload size

Never commit `.env`.

Maintain `.env.example`.

---

# Testing

Every major feature must have tests.

At minimum:

- unit tests
- API tests
- ingestion tests
- retrieval tests
- integration tests
- end-to-end tests

External LLM/model APIs must be mocked in normal unit tests.

Do not require external API access for the test suite.

---

# Development Workflow

Before modifying code:

1. Inspect the repository.
2. Read applicable instruction files.
3. Understand existing architecture.
4. Identify existing implementations.
5. Avoid rewriting working code.

After modifying code:

1. Run relevant tests.
2. Fix failures.
3. Check imports.
4. Check type consistency.
5. Verify API contracts.
6. Update documentation when appropriate.

Do not implement unrelated features while working on a phase.

---

# Definition of Done

A feature is not complete until:

- implementation exists
- tests exist
- tests pass
- errors are handled
- configuration is documented
- API contracts are documented when applicable
- existing functionality continues to work

Never claim a feature is complete merely because the code was written.