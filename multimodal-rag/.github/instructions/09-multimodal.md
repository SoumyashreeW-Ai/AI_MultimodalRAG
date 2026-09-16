# Multimodal RAG Instructions

## Definition

This project must implement actual multimodal RAG.

The system must support both:

text retrieval

and

image retrieval.

---

# Image Pipeline

For each image:

1. Extract/save image.
2. Validate image.
3. Generate image embedding.
4. Store image vector.
5. Store image metadata.
6. Optionally generate caption.
7. Store caption.
8. Make caption searchable when available.

---

# Query Pipeline

For a user query:

TEXT PATH:

query
→ text embedding
→ text_chunks
→ text results

IMAGE PATH:

query
→ CLIP/multimodal text embedding
→ images
→ image results

Then:

text results
+
image results
→ score normalization
→ fusion
→ reranking
→ multimodal context

---

# Vision LLM

When supported:

Send:

- user question
- retrieved text
- retrieved images
- image metadata

to the multimodal LLM.

The model should be able to reason about visual information.

---

# Fallback

If the LLM does not support image inputs:

image
→ caption
→ text embedding
→ retrieval/context

This allows the application to remain functional.

---

# Image Metadata

Never lose:

- document
- page
- image ID
- image path

This information is required for citations and UI rendering.

---

# Streamlit

The UI must show relevant retrieved images with:

- filename
- page
- relevance score where appropriate

---

# Important

Do not implement a fake multimodal pipeline where:

image retrieval happens

but

the image is never used by the answer generation system.