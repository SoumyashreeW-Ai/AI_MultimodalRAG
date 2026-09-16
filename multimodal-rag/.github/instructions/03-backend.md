# FastAPI Backend Instructions

## General

Use FastAPI for all backend APIs.

Use Pydantic models for requests and responses.

Routes should be thin.

Routes should call application/service functions.

---

# Upload

POST /api/documents/upload

The endpoint must:

1. Validate extension.
2. Validate MIME type where possible.
3. Validate file size.
4. Save file safely.
5. Calculate SHA-256.
6. Check duplicate.
7. Start ingestion.
8. Return ingestion result.

Never trust the original filename as a filesystem path.

---

# Query

POST /api/query

Request:

{
  "query": "...",
  "top_k": 5
}

Response should contain:

- answer
- sources
- images
- retrieval information where debug mode is enabled

---

# Errors

Use appropriate HTTP status codes.

Do not expose:

- stack traces
- API keys
- internal filesystem details
- implementation secrets

Log detailed errors server-side.

Return useful client-safe messages.

---

# Health

GET /api/health

Must verify:

- application is running
- ChromaDB is accessible

Do not make an external LLM request simply to determine basic application health.

---

# Dependency Injection

Use FastAPI dependency injection for services where useful.

Services should be easy to replace with mocks during testing.