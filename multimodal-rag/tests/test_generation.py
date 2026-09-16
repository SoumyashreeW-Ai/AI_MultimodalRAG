from types import SimpleNamespace

from app.core import config as config_module
from app.generation.llm import MockLLMProvider as RealMockLLMProvider, OpenAICompatibleProvider
from app.generation.service import RAGGenerationService, _clean_answer_text
from app.llm.providers import LLMProvider


class MockLLMProvider(LLMProvider):
    def __init__(self, supports_images: bool = False):
        self._supports = supports_images
        self.last_call = {}

    @property
    def supports_image_inputs(self) -> bool:
        return self._supports

    def load_model(self):
        return self

    def generate(self, system_prompt, user_prompt, images=None):
        # record call for assertions
        self.last_call = {"system": system_prompt, "user": user_prompt, "images": images}
        # return a simple grounded text
        return {"text": "Answer based on provided context."}


def _make_text_hit():
    return {
        "id": "c1",
        "score": 0.9,
        "content_type": "text",
        "document_id": "doc1",
        "filename": "a.txt",
        "page": 1,
        "chunk_id": "c1",
        "text": "Alpha text",
        "metadata": {"document_id": "doc1", "filename": "a.txt", "page": 1, "chunk_id": "c1"},
    }


def _make_image_hit(path="/tmp/i.png", caption=None):
    return {
        "id": "img1",
        "score": 0.8,
        "content_type": "image",
        "document_id": "doc2",
        "filename": "i.png",
        "page": 2,
        "image_path": path,
        "metadata": {"document_id": "doc2", "filename": "i.png", "page": 2, "image_id": "img1", "image_path": path, "caption": caption},
    }


def test_generation_sends_images_when_supported():
    llm = MockLLMProvider(supports_images=True)
    svc = RAGGenerationService(llm)
    text_hit = _make_text_hit()
    image_hit = _make_image_hit(path="/tmp/img.png", caption="A cat")

    answer = svc.generate("What is this?", [text_hit], [image_hit])

    assert answer.answer == "Answer based on provided context."
    assert answer.images == ["/tmp/img.png"]
    assert llm.last_call["images"] == ["/tmp/img.png"]


def test_generation_uses_captions_when_images_not_supported():
    llm = MockLLMProvider(supports_images=False)
    svc = RAGGenerationService(llm)
    text_hit = _make_text_hit()
    image_hit = _make_image_hit(path="/tmp/img2.png", caption="A dog")

    answer = svc.generate("Describe image", [text_hit], [image_hit])

    assert answer.answer == "Answer based on provided context."
    assert answer.images == []
    assert llm.last_call["images"] is None
    assert "A dog" in llm.last_call["user"]


def test_mock_llm_uses_image_descriptions_for_image_only_contexts():
    llm = RealMockLLMProvider(vision=True)
    prompt = (
        "USER QUESTION:\nDescribe the photo\n\n"
        "TEXT SOURCES:\n\n"
        "IMAGE SOURCES:\n"
        "Image 1:\nfilename: photo.png\npage: 1\nimage description: Image 30x30 pixels; "
        "dominant color multicolored; no OCR text detected in the source image.\nimage: [image omitted]\n\n"
        "Please answer concisely and cite which sources you used."
    )

    answer = llm.generate("system", prompt, images=["/tmp/photo.png"])

    assert "30x30" in answer["text"]
    assert "multicolored" in answer["text"].lower()


def test_mock_llm_prefers_current_image_description_over_stale_text_sources():
    llm = RealMockLLMProvider(vision=True)
    prompt = (
        "USER QUESTION:\nDescribe the photo\n\n"
        "TEXT SOURCES:\nSource 1: Payment Service communicates with PostgreSQL.\n\n"
        "IMAGE SOURCES:\nImage 1:\nfilename: photo.png\npage: 1\nimage description: "
        "Image 30x30 pixels; dominant color multicolored; no OCR text detected in the source image.\nimage: [image omitted]\n\n"
        "Please answer concisely and cite which sources you used."
    )

    answer = llm.generate("system", prompt, images=["/tmp/photo.png"])

    assert "Payment Service" not in answer["text"]
    assert "30x30" in answer["text"]
    assert "multicolored" in answer["text"].lower()


def test_mock_llm_prefers_text_source_for_factual_questions():
    llm = RealMockLLMProvider(vision=True)
    prompt = (
        "USER QUESTION:\nWhat does the payment service do?\n\n"
        "TEXT SOURCES:\nSource 1: Payment Service communicates with PostgreSQL.\n\n"
        "IMAGE SOURCES:\nImage 1:\nfilename: photo.png\npage: 1\nimage description: "
        "Image 30x30 pixels; dominant color multicolored; no OCR text detected in the source image.\nimage: [image omitted]\n\n"
        "Please answer concisely and cite which sources you used."
    )

    answer = llm.generate("system", prompt, images=["/tmp/photo.png"])

    assert "Payment Service communicates with PostgreSQL." in answer["text"]
    assert "30x30" not in answer["text"]


def test_clean_answer_text_strips_raw_source_dump():
    raw_answer = (
        "Source 1: Payment Service communicates with PostgreSQL.\n"
        "TEXT SOURCES:\nPayment Service communicates with PostgreSQL.\n"
        "IMAGE SOURCES:\nImage 1: photo.png\n"
        "Please answer concisely and cite which sources you used."
    )

    cleaned = _clean_answer_text(raw_answer)

    assert "Payment Service communicates with PostgreSQL." in cleaned
    assert "TEXT SOURCES:" not in cleaned
    assert "Source 1:" not in cleaned
    assert "Please answer concisely" not in cleaned


def test_generation_service_falls_back_to_mock_when_provider_fails():
    class FailingProvider:
        supports_image_inputs = False

        def generate(self, system_prompt, user_prompt, images=None):
            raise RuntimeError("LLM request failed (401): invalid API key")

    svc = RAGGenerationService(FailingProvider())
    answer = svc.generate(
        "Who talks to DB?",
        [{"filename": "a.txt", "page": 1, "text": "Payment Service communicates with PostgreSQL."}],
        [],
    )

    assert "Payment Service" in answer.answer
    assert "PostgreSQL" in answer.answer


def test_openai_compatible_provider_uses_chat_messages_and_extracts_text(monkeypatch):
    provider = OpenAICompatibleProvider()
    provider.settings = SimpleNamespace(
        llm_api_key="sk-test-key",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        llm_max_tokens=256,
        llm_base_url="https://example.test/v1/chat/completions",
        vision_enabled=False,
        upload_dir="/tmp",
    )

    sent = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Grounded answer from the model."}}]}

    def fake_post(url, json, headers, timeout):
        sent["url"] = url
        sent["json"] = json
        sent["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("app.generation.llm.httpx.post", fake_post)

    result = provider.generate("System instruction", "User question")

    assert result["text"] == "Grounded answer from the model."
    assert sent["json"]["messages"][0]["role"] == "system"
    assert sent["json"]["messages"][1]["content"] == "User question"
    assert sent["headers"]["Authorization"].startswith("Bearer ")


def test_settings_exposes_llm_api_key_as_truthy_when_present(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    config_module._settings = None

    settings = config_module.get_settings()

    assert bool(settings.llm_api_key) is True
