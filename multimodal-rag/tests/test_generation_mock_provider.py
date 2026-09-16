from fastapi.testclient import TestClient
from app.generation.llm import MockLLMProvider
from app.generation.service import RAGGenerationService
from app.main import app


def test_mock_provider_direct():
    mock = MockLLMProvider(vision=True)
    svc = RAGGenerationService(mock)
    text_hit = {"filename": "a.txt", "page": 1, "text": "Payment Service details"}
    image_hit = {"filename": "i.png", "page": 1, "image_path": "data/uploads/i.png", "metadata": {"caption": "diagram"}}
    resp = svc.generate("Who talks to DB?", [text_hit], [image_hit])
    assert resp.answer == "The Payment Service communicates with PostgreSQL."


def test_mock_provider_for_audio_context():
    mock = MockLLMProvider(vision=False)
    user_prompt = "USER QUESTION:\nWhat is in this audio?\n\nTEXT SOURCES:\nSource 1:\nThis audio clip contains silence.\n\nPlease answer concisely and cite which sources you used."
    resp = mock.generate("system", user_prompt)
    assert "silence" in resp["text"].lower()
    assert "audio" in resp["text"].lower()


def test_mock_provider_handles_missing_transcript_naturally():
    mock = MockLLMProvider(vision=False)
    user_prompt = "USER QUESTION:\nWhen was the Apollo project launched?\n\nTEXT SOURCES:\nSource 1:\n[transcription unavailable] file: 1234-demo.wav\n\nPlease answer concisely and cite which sources you used."
    resp = mock.generate("system", user_prompt)
    text = resp["text"]
    assert "transcript" in text.lower()
    assert "file:" not in text.lower()
    assert "apollo" not in text.lower() or "The transcript" in text


def test_generation_service_strips_prompt_wrapper_from_answer():
    class FakeLLM:
        supports_image_inputs = False

        def generate(self, system_prompt, user_prompt, images=None):
            return {"text": "Answer: The Payment Service communicates with PostgreSQL. Source 1: a.txt"}

    svc = RAGGenerationService(FakeLLM())
    result = svc.generate("Who talks to DB?", [{"filename": "a.txt", "page": 1, "text": "Payment Service communicates with PostgreSQL", "metadata": {"filename": "a.txt", "page": 1, "content_type": "text"}}], [])
    assert result.answer == "The Payment Service communicates with PostgreSQL."
    assert "Source 1" not in result.answer


def test_query_endpoint_with_mock(monkeypatch):
    import app.main as mainmod

    def fake_query(q, top_k=None, debug=False):
        return {"results": [{"id": "c1", "score": 0.9, "content_type": "text", "text": "Payment Service communicates with PostgreSQL", "metadata": {"filename": "a.txt", "page": 1}} , {"id": "i1", "score": 0.8, "content_type": "image", "image_path": "data/uploads/i.png", "metadata": {"filename": "i.png", "page": 1, "caption": "diagram"}} ]}

    class FakeRetrieval:
        def query(self, q, top_k=None, debug=False):
            return fake_query(q, top_k=top_k, debug=debug)

    monkeypatch.setattr(mainmod, "retrieval_service", FakeRetrieval())

    # Patch generation_service to use MockLLMProvider returning fixed text
    mock = MockLLMProvider(vision=True)

    def fake_generate(query, text_results, image_results=None):
        class R:
            answer = "The Payment Service communicates with PostgreSQL."
            sources = []
            images = ["data/uploads/i.png"]
            retrieval_debug = None

        return R()

    class FakeGeneration:
        def generate(self, query, text_results, image_results=None):
            return fake_generate(query, text_results, image_results)

    monkeypatch.setattr(mainmod, "generation_service", FakeGeneration())

    # Call the endpoint function directly to avoid TestClient semantics
    from app.api import schemas

    req = schemas.QueryRequest(query="Who talks to DB?", top_k=3)
    resp = mainmod.query_endpoint(req)
    assert resp.answer == "The Payment Service communicates with PostgreSQL."
    assert resp.images
