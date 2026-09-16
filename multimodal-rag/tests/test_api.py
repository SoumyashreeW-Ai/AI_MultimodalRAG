from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "ok" in r.json()["status"]


def test_list_documents(monkeypatch):
    # monkeypatch vectorstore.get_document_count
    import app.main as mainmod

    monkeypatch.setattr(mainmod.vectorstore, "get_document_count", lambda: 42)
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json()["count"] == 42


def test_delete_document_not_found(monkeypatch):
    import app.main as mainmod

    monkeypatch.setattr(mainmod.vectorstore, "delete_document", lambda doc: False)
    r = client.delete("/api/documents/some-id")
    assert r.status_code == 404


def test_delete_document_success(monkeypatch):
    import app.main as mainmod

    monkeypatch.setattr(mainmod.vectorstore, "delete_document", lambda doc: True)
    r = client.delete("/api/documents/some-id")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"


def test_reindex_calls_indexing(monkeypatch, tmp_path):
    import app.main as mainmod

    def fake_index(path, document_id=None, storage_dir=None):
        return {"document_id": document_id or "doc-x", "filename": "f", "status": "indexed"}

    monkeypatch.setattr(mainmod, "indexing_service", mainmod.indexing_service)
    monkeypatch.setattr(mainmod.indexing_service, "index_document", fake_index)

    r = client.post("/api/reindex/doc-1", json={"file_path": "/tmp/x.pdf"})
    assert r.status_code == 200
    assert r.json()["status"] == "indexed"


def test_query_endpoint(monkeypatch):
    import app.main as mainmod

    # Mock retrieval_service.query
    def fake_query(q, top_k=None, debug=False):
        return {"results": [{"id": "c1", "score": 0.9, "content_type": "text", "text": "Alpha", "metadata": {"filename": "a.txt", "page": 1}}]}

    class FakeRAG:
        def __init__(self):
            self.answer = "ans"
            self.sources = []
            self.images = []
            self.retrieval_debug = None

    def fake_generate(query, text_results, image_results=None):
        return FakeRAG()

    monkeypatch.setattr(mainmod, "retrieval_service", mainmod.retrieval_service)
    monkeypatch.setattr(mainmod.retrieval_service, "query", fake_query)
    monkeypatch.setattr(mainmod, "generation_service", mainmod.generation_service)
    monkeypatch.setattr(mainmod.generation_service, "generate", fake_generate)

    r = client.post("/api/query", json={"query": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "ans"
    assert "retrieval_debug" not in r.json()


def test_query_endpoint_includes_safe_retrieval_debug_when_enabled(monkeypatch):
    import app.main as mainmod

    captured = {}

    def fake_query(q, top_k=None, debug=False):
        captured["debug"] = debug
        return {
            "results": [],
            "debug": {
                "text_hits": [
                    {
                        "id": "chunk-1",
                        "score": 0.21,
                        "text": "x" * 600,
                        "metadata": {"chunk_id": "chunk-1", "filename": "architecture.pdf", "page": 12},
                    }
                ],
                "image_hits": [
                    {
                        "id": "image-1",
                        "score": 0.24,
                        "metadata": {
                            "image_id": "image-1",
                            "filename": "architecture.pdf",
                            "page": 98,
                            "caption": "Payment architecture diagram",
                            "image_path": "/tmp/not-safe.png",
                        },
                    }
                ],
            },
        }

    class FakeRAG:
        answer = "ans"
        sources = []
        images = []

    monkeypatch.setenv("DEBUG_RAG", "true")
    monkeypatch.setattr(mainmod.retrieval_service, "query", fake_query)
    monkeypatch.setattr(mainmod.generation_service, "generate", lambda *args: FakeRAG())

    r = client.post("/api/query", json={"query": "What database does the Payment Service use?"})

    assert r.status_code == 200
    assert captured["debug"] is True
    debug = r.json()["retrieval_debug"]
    assert debug["query"] == "What database does the Payment Service use?"
    assert debug["text_results"][0]["text_preview"] == "x" * 500
    assert debug["image_results"][0]["caption"] == "Payment architecture diagram"
    assert "image_path" not in debug["image_results"][0]


def test_build_default_text_embedding_provider_uses_fallback_when_sentence_transformers_is_unavailable(monkeypatch):
    import builtins
    from types import SimpleNamespace

    import app.embeddings.providers as providers

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = providers.build_default_text_embedding_provider(SimpleNamespace(text_embedding_model="all-MiniLM-L6-v2"))
    vectors = provider.embed_texts(["hello world"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 128


def test_vectorstore_counts_unique_documents_only(monkeypatch):
    from app.vectorstore.collections import ChromaVectorStore

    class FakeCollection:
        def get(self, include=None):
            return {
                "metadatas": [
                    {"document_id": "doc-1"},
                    {"document_id": "doc-1"},
                    {"document_id": "doc-2"},
                ]
            }

    vectorstore = ChromaVectorStore.__new__(ChromaVectorStore)
    vectorstore.collections = {"text_chunks": FakeCollection(), "images": FakeCollection()}

    assert vectorstore.get_document_count() == 2
