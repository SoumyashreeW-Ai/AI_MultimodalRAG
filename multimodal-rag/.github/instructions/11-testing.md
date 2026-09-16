# Testing Instructions

## Unit Tests

Test independently:

- configuration
- hashing
- chunking
- PDF parsing
- DOCX parsing
- image processing
- embeddings using mocks
- ChromaDB repository
- retrieval
- fusion
- generation using mocks

---

# API Tests

Use FastAPI TestClient/httpx.

Test:

POST /api/documents/upload

GET /api/documents

DELETE /api/documents/{document_id}

POST /api/query

GET /api/health

---

# Integration Tests

Test:

document

→ parser

→ chunker

→ embeddings

→ ChromaDB

→ retrieval

Use lightweight/mock embedding providers where necessary.

---

# End-to-End

Create at least one complete E2E test:

upload

→ ingestion

→ indexing

→ query

→ retrieval

→ generation

→ response

---

# External Services

Do not require real external API calls for ordinary tests.

Use dependency injection and mocks.

---

# Regression

Whenever a bug is fixed, add a regression test.