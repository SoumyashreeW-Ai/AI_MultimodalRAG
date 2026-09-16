# LLM Generation Instructions

## Grounding

The LLM must answer using retrieved context.

The system prompt must instruct the model:

- do not invent information
- do not use unsupported facts
- say when the answer is not present
- cite sources
- distinguish text-derived and image-derived information

---

# Context

Construct explicit context containing:

TEXT SOURCES

IMAGE SOURCES

SOURCE METADATA

USER QUESTION

---

# Sources

Every answer should have structured source references.

A source should contain:

- filename
- page
- content_type
- chunk_id where applicable
- retrieval score where useful

---

# Provider Abstraction

Do not couple RAG logic to a specific LLM vendor.

Use an LLM provider abstraction.

---

# Multimodal Models

If the selected model accepts images:

Send relevant retrieved images to the model.

If it does not:

Use image captions/descriptions as text context.

---

# Failure

If the LLM cannot answer from retrieved context:

Return a grounded response stating that the information was not found.

Do not fabricate an answer.