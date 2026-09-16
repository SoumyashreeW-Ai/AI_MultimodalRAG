# Retrieval Instructions

## Query Flow

For every user query:

1. Validate query.
2. Generate text query embedding.
3. Search text collection.
4. Generate multimodal/CLIP query embedding.
5. Search image collection.
6. Normalize scores.
7. Fuse results.
8. Deduplicate where appropriate.
9. Optionally rerank.
10. Return final context.

---

# Text Retrieval

Retrieve more candidates than the final answer requires.

Example:

top_k = 5

candidate_k = 20

Then rerank/fuse down to 5.

---

# Image Retrieval

Image retrieval must use a text-to-image compatible embedding space.

Do not perform meaningless comparison between unrelated embedding spaces.

---

# Fusion

Text and image scores may have different distributions.

Normalize before combining.

Make weights configurable.

Example:

text_weight

image_weight

---

# RetrievalResult

Use a typed RetrievalResult model.

Fields should include:

- id
- score
- content_type
- document_id
- filename
- page
- chunk_id
- text
- image_path
- metadata

---

# Debugging

Support optional retrieval debugging.

Debug information may include:

- retrieved IDs
- scores
- source files
- pages
- content types

Do not expose excessive internal information in normal production responses.