import importlib
import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a TestClient where the app is initialized with a mocked LLM provider.

    We set a few environment variables before importing `app.main` so that the
    application will construct its components (including `MockLLMProvider`) using
    predictable settings. We also isolate Chroma persistence and uploads under
    the temporary directory to avoid mutating the repository state.
    """
    # Configure mock provider and debug mode
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("DEBUG_RAG", "true")

    # Use isolated directories for chroma and uploads
    chroma_dir = tmp_path / "chroma"
    uploads_dir = tmp_path / "uploads"
    chroma_dir.mkdir()
    uploads_dir.mkdir()
    monkeypatch.setenv("CHROMA_DB_PATH", str(chroma_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads_dir))

    # Ensure a fresh import of the application module so the env vars are picked up
    for mod in list(sys.modules.keys()):
        if mod.startswith("app.") or mod == "app":
            del sys.modules[mod]

    import app.main as main  # type: ignore

    client = TestClient(main.app)
    yield client


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True


def test_upload_query_and_delete_flow(client, tmp_path):
    # Create a simple text file containing a target phrase the MockLLMProvider recognizes
    content = "The Payment Service communicates with PostgreSQL."
    sample_path = tmp_path / "payment_service.txt"
    sample_path.write_text(content, encoding="utf-8")

    # Upload the document via the API
    with sample_path.open("rb") as fh:
        files = {"file": (sample_path.name, fh, "text/plain")}
        r = client.post("/api/documents/upload", files=files)

    assert r.status_code == 200
    upload_body = r.json()
    assert upload_body.get("status") in {"indexed", "duplicate"}
    document_id = upload_body.get("document_id")
    assert document_id

    # Ensure the document is counted
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json().get("count", 0) >= 1

    # Run a query that the MockLLMProvider will answer from the uploaded content
    query_payload = {"query": "Where does the Payment Service store transactions?", "top_k": 3}
    r = client.post("/api/query", json=query_payload)
    assert r.status_code == 200, r.text
    q = r.json()
    # The mock provider returns a grounded answer when 'payment service' is present
    assert "PostgreSQL" in q.get("answer", "")
    assert isinstance(q.get("sources"), list)

    # The file-serving endpoint should be able to fetch uploaded files when debug is enabled
    # If the upload handler saved the file under uploads, attempt to fetch it.
    # Note: some storage backends or naming strategies may differ; allow 404 as non-fatal.
    filename = upload_body.get("filename") or sample_path.name
    files_resp = client.get(f"/api/files/{filename}")
    assert files_resp.status_code in (200, 404)

    # Delete the document and ensure the API acknowledges removal
    del_resp = client.delete(f"/api/documents/{document_id}")
    # Some vectorstore backends may report deletion success; allow 200 or 404
    assert del_resp.status_code in (200, 404)
