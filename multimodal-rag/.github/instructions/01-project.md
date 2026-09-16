# Project Requirements

## Goal

Build an end-to-end multimodal RAG application capable of ingesting documents and images and answering questions using retrieved text and images.

---

# Supported Files

The initial implementation must support:

- PDF
- DOCX
- TXT
- Markdown
- PNG
- JPG
- JPEG

The architecture should allow future file types to be added without rewriting the ingestion pipeline.

---

# Functional Requirements

The application must support:

1. Upload documents.
2. Calculate document SHA-256.
3. Detect duplicate documents.
4. Parse documents.
5. Extract text.
6. Extract embedded images.
7. Chunk text.
8. Generate text embeddings.
9. Generate image embeddings.
10. Store vectors in ChromaDB.
11. Search text.
12. Search images.
13. Combine retrieval results.
14. Optionally rerank results.
15. Generate grounded answers.
16. Return source information.
17. Return retrieved images.
18. List indexed documents.
19. Delete documents.
20. Reindex documents.
21. Provide health checks.
22. Provide Streamlit UI.

---

# Required Metadata

Every text chunk must contain:

- document_id
- filename
- source_path
- page
- chunk_id
- content_type

Every image must contain:

- document_id
- filename
- source_path
- page
- image_id
- image_path
- content_type

If available, images should also contain:

- caption
- image_width
- image_height

---

# API

Required endpoints:

GET /api/health

POST /api/documents/upload

GET /api/documents

DELETE /api/documents/{document_id}

POST /api/reindex/{document_id}

POST /api/query

---

# Frontend

Streamlit must provide:

- document upload
- indexed document list
- delete action
- chat interface
- answer display
- source display
- retrieved image display
- backend error handling
- loading states