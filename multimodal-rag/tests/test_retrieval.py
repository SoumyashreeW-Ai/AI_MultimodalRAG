from app.models.retrieval_models import RetrievalResult
from app.retrieval.service import RetrievalService


class MockTextEmbeddingProvider:
    def embed_text(self, text):
        return [1.0, 0.0]


class MockMultimodalEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def embed_text(self, text):
        self.calls.append(("query", text))
        return [0.0, 1.0]


class MockVectorStore:
    def query_text(self, query_vector, n_results):
        return [
            {
                "id": "chunk-1",
                "score": 0.9,
                "document_id": "doc-1",
                "filename": "a.txt",
                "page": 1,
                "chunk_id": "chunk-1",
                "text": "Alpha",
                "metadata": {"document_id": "doc-1", "filename": "a.txt", "page": 1, "chunk_id": "chunk-1"},
            },
            {
                "id": "chunk-2",
                "score": 0.4,
                "document_id": "doc-2",
                "filename": "b.txt",
                "page": 2,
                "chunk_id": "chunk-2",
                "text": "Beta",
                "metadata": {"document_id": "doc-2", "filename": "b.txt", "page": 2, "chunk_id": "chunk-2"},
            },
        ]

    def query_images(self, query_vector, n_results):
        return [
            {
                "id": "img-1",
                "score": 0.8,
                "document_id": "doc-3",
                "filename": "c.png",
                "page": 3,
                "image_path": "/tmp/c.png",
                "metadata": {"document_id": "doc-3", "filename": "c.png", "page": 3, "image_id": "img-1", "image_path": "/tmp/c.png"},
            }
        ]


def test_query_returns_fused_results_and_deduplicates():
    store = MockVectorStore()
    multimodal_provider = MockMultimodalEmbeddingProvider()
    service = RetrievalService(
        vectorstore=store,
        text_embedding_provider=MockTextEmbeddingProvider(),
        multimodal_embedding_provider=multimodal_provider,
        text_weight=1.0,
        image_weight=1.0,
        top_k=5,
        candidate_k=10,
        rerank=False,
    )

    payload = service.query("some query", top_k=3)

    assert len(payload["results"]) == 3
    assert {item["content_type"] for item in payload["results"]} == {"text", "image"}
    assert payload["results"][0]["score"] >= payload["results"][1]["score"]
    # chunk-2 has the lower cosine distance (0.4), so it is now ranked first.
    assert payload["results"][0]["document_id"] == "doc-2"
    assert multimodal_provider.calls == [("query", "some query")]


def test_query_validates_empty_input():
    service = RetrievalService(
        vectorstore=MockVectorStore(),
        text_embedding_provider=MockTextEmbeddingProvider(),
        multimodal_embedding_provider=MockMultimodalEmbeddingProvider(),
    )

    try:
        service.query("   ")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_retrieval_result_model_fields():
    result = RetrievalResult(
        id="chunk-1",
        score=0.75,
        content_type="text",
        document_id="doc-1",
        filename="a.txt",
        page=1,
        chunk_id="chunk-1",
        text="Alpha",
        metadata={"source": "mock"},
    )

    assert result.content_type == "text"
    assert result.metadata["source"] == "mock"


def test_cosine_distance_is_converted_to_descending_similarity():
    service = RetrievalService(
        vectorstore=MockVectorStore(),
        text_embedding_provider=MockTextEmbeddingProvider(),
        multimodal_embedding_provider=MockMultimodalEmbeddingProvider(),
    )
    hits = service._normalize_scores([{"score": 0.5}, {"score": 0.1}])

    assert hits[0]["similarity_score"] == 0.5
    assert hits[1]["similarity_score"] == 0.9
    assert hits[1]["similarity_score"] > hits[0]["similarity_score"]


def test_best_chroma_distance_remains_best_after_retrieval_service_processing():
    class DistanceVectorStore:
        def query_text(self, query_vector, n_results):
            # Deliberately unordered to verify RetrievalService performs the
            # descending similarity ordering itself.
            return [
                {"id": "far", "score": 0.5, "text": "Far", "metadata": {}},
                {"id": "near", "score": 0.1, "text": "Near", "metadata": {}},
            ]

        def query_images(self, query_vector, n_results):
            return []

    service = RetrievalService(
        vectorstore=DistanceVectorStore(),
        text_embedding_provider=MockTextEmbeddingProvider(),
        multimodal_embedding_provider=MockMultimodalEmbeddingProvider(),
        top_k=2,
    )

    results = service.query("query", top_k=2)["results"]

    assert [result["id"] for result in results] == ["near", "far"]
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.5
