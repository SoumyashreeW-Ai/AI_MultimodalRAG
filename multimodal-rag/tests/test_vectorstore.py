import pytest

from app.vectorstore.collections import ChromaVectorStore


@pytest.fixture
def vectorstore(tmp_path):
    store = ChromaVectorStore(persist_directory=str(tmp_path / "chroma"))
    return store


def test_add_query_and_delete_text_chunks(vectorstore):
    text_chunks = [
        {
            "id": "doc-1:page-1:chunk-1",
            "text": "Alpha beta gamma",
            "metadata": {
                "document_id": "doc-1",
                "filename": "sample.pdf",
                "page": 1,
                "chunk_id": "chunk-1",
                "source_path": "/tmp/sample.pdf",
                "content_type": "text",
            },
        }
    ]

    inserted = vectorstore.add_text_chunks(text_chunks, embeddings=[[0.1, 0.2, 0.3]])
    assert inserted == 1
    assert vectorstore.document_exists("doc-1") is True

    results = vectorstore.query_text(query_vector=[0.1, 0.2, 0.3], n_results=5)
    assert len(results) >= 1
    assert results[0]["document_id"] == "doc-1"

    deleted = vectorstore.delete_document("doc-1")
    assert deleted is True
    assert vectorstore.document_exists("doc-1") is False


def test_add_query_images_and_health_check(vectorstore):
    images = [
        {
            "id": "doc-2:page-1:image-1",
            "metadata": {
                "document_id": "doc-2",
                "filename": "chart.png",
                "page": 1,
                "image_id": "image-1",
                "image_path": "/tmp/chart.png",
                "content_type": "image/png",
                "caption": "A chart",
            },
        }
    ]

    inserted = vectorstore.add_images(images, embeddings=[[0.4, 0.5, 0.6]])
    assert inserted == 1
    assert vectorstore.health_check() is True

    results = vectorstore.query_images(query_vector=[0.4, 0.5, 0.6], n_results=5)
    assert len(results) >= 1
    assert results[0]["document_id"] == "doc-2"


def test_custom_text_collection_upserts_without_opening_images(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        text_collection_name="text_chunks_v2",
        include_images=False,
    )
    record = {
        "id": "doc-1:page-1:chunk-1",
        "text": "Updated text",
        "metadata": {"document_id": "doc-1", "content_type": "text"},
    }

    assert store.text_collection_name == "text_chunks_v2"
    assert "images" not in store.collections
    assert store.upsert_text_chunks([record], [[0.1, 0.2, 0.3]]) == 1
    assert store.upsert_text_chunks([record], [[0.1, 0.2, 0.3]]) == 1
    assert store.collections["text_chunks"].count() == 1


def test_metadata_with_none_values_is_stripped_before_chroma_insert(vectorstore):
    record = {
        "id": "doc-3:page-1:chunk-1",
        "text": "Alpha beta",
        "metadata": {
            "document_id": "doc-3",
            "filename": "sample.txt",
            "page": 1,
            "chunk_id": "chunk-1",
            "source_path": "/tmp/sample.txt",
            "content_type": "text",
            "start_time": None,
            "end_time": None,
        },
    }

    inserted = vectorstore.add_text_chunks([record], embeddings=[[0.1, 0.2, 0.3]])
    assert inserted == 1
    stored = vectorstore.collections["text_chunks"].get(include=["metadatas"])
    assert stored["metadatas"][0]["document_id"] == "doc-3"
    assert "start_time" not in stored["metadatas"][0]
    assert "end_time" not in stored["metadatas"][0]
