# Ingestion Pipeline

## Principle

Separate:

Parsing

from:

Chunking

from:

Embedding

from:

Storage

Do not create one giant ingestion function.

---

# PDF

Use a reliable PDF parser.

Extract text page-by-page.

Preserve page number.

Extract embedded images where technically possible.

Each extracted image must retain:

- document_id
- filename
- page
- image_id

---

# DOCX

Extract:

- paragraphs
- useful structural information
- embedded images

Preserve document metadata where useful.

---

# TXT / Markdown

Read text safely using explicit encoding handling.

Handle empty files.

---

# Images

Validate image format.

Use Pillow for validation and normalization.

Save extracted images with deterministic names.

Do not allow path traversal.

---

# Failure Handling

A corrupted image should not necessarily cause unrelated text ingestion to fail.

A missing image caption should not make the complete document unusable.

Record ingestion errors clearly.

---

# Idempotency

Use document SHA-256 to detect duplicates.

Do not create duplicate vectors when the same document is uploaded again.

Support explicit reindexing.