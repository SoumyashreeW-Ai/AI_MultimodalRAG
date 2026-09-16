# ChromaDB Instructions

## Collections

Use separate collections initially:

text_chunks

images

---

# Text Record

Each text record should include:

id

document/chunk text

metadata:

- document_id
- filename
- page
- chunk_id
- source_path
- content_type

---

# Image Record

Each image record should include:

id

embedding

metadata:

- document_id
- filename
- page
- image_id
- image_path
- content_type
- caption when available

---

# IDs

IDs must be deterministic.

Do not use random IDs when deterministic IDs can be generated.

Example:

document_id + page + chunk_id

---

# Persistence

Use persistent ChromaDB storage.

The path must come from configuration.

---

# Operations

Provide methods for:

add_text_chunks

add_images

query_text

query_images

delete_document

document_exists

get_document_count

health_check

---

# Abstraction

Application code should not depend directly on ChromaDB internals.

Wrap ChromaDB in a repository/service abstraction.

---

# Testing

Use temporary ChromaDB directories for tests.