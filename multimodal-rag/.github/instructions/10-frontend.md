# Streamlit Instructions

## Architecture

Streamlit is a client/UI layer.

It must communicate with FastAPI.

Do not put:

- ChromaDB logic
- embedding logic
- ingestion logic
- LLM logic

inside Streamlit.

---

# UI

Provide:

Sidebar:

- backend URL
- file uploader
- indexed documents
- delete controls

Main:

- chat history
- question input
- answer
- sources
- retrieved images

---

# Session State

Use Streamlit session_state for chat history.

Do not store application-wide mutable state in module globals.

---

# Error Handling

Handle:

- backend unavailable
- timeout
- invalid upload
- ingestion failure
- query failure

Display user-friendly messages.

---

# Images

Display retrieved images using the API-provided image information.

Do not expose arbitrary server filesystem paths to the browser.