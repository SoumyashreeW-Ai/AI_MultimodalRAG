(Multimodal RAG) Project
-----------------------

Environment variables (add to `.env`):

- `LLM_PROVIDER` (e.g. `openai` or `mock`)
- `LLM_API_KEY` (your provider API key)
- `LLM_MODEL` (model name)
- `LLM_BASE_URL` (optional custom base URL)
- `LLM_TEMPERATURE` (float)
- `LLM_MAX_TOKENS` (int)
- `VISION_ENABLED` (true/false)

Run backend:

```bash
uvicorn app.main:app --reload
```

Run frontend:

```bash
streamlit run frontend/streamlit_app.py
```

Observability (Comet / Opik)
-----------------------------

This project includes lightweight instrumentation for Comet/Opik. To enable it:

- Install the SDK:

```bash
pip install comet-ml
```

- Set your Comet API key in the environment (or add to `.env`):

```bash
export COMET_API_KEY="<your-comet-api-key>"
# or add COMET_API_KEY=<your-comet-api-key> to .env
```

- Restart the backend. The app initializes Opik at startup and will emit metrics for:
	- embedding times and batch sizes during indexing
	- retrieval latency and top-hit metadata
	- LLM latency, status codes, and retry flags

Notes
- The instrumentation logs metadata and timings only; it avoids logging full document text or secrets by default. Do not send PII or API keys to Comet unless you explicitly accept the privacy implications.
- View events and metrics in the Comet/Opik dashboard under the `multimodal-rag` project.


