# Production Readiness

## Logging

Use structured logging.

Log:

- request ID
- document ID
- query ID
- ingestion status
- retrieval status
- errors

Do not log:

- API keys
- secrets
- sensitive user content unnecessarily

---

# Observability

Make it possible to measure:

- ingestion duration
- embedding duration
- retrieval duration
- LLM duration
- number of retrieved text chunks
- number of retrieved images

---

# Configuration

All production-sensitive configuration must come from environment variables.

---

# Docker

Support:

FastAPI

Streamlit

persistent data

environment variables

health checks

---

# Performance

Avoid:

- loading embedding models per request
- unnecessary repeated file reads
- unnecessary repeated embeddings
- unbounded document uploads

Use batching where appropriate.

---

# Maintainability

Prefer simple working architecture over unnecessary abstractions.

Do not introduce distributed systems or complex infrastructure unless required.