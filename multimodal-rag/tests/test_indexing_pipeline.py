from pathlib import Path

from PIL import Image

from app.embeddings.providers import DefaultImageCaptionProvider
from app.indexing.service import IndexingService
from app.vectorstore.collections import ChromaVectorStore


class MockTextEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [[float(len(text)), 0.1] for text in texts]


class MockImageEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed_images(self, image_paths):
        self.calls += 1
        return [[float(len(path)), 0.2] for path in image_paths]


def test_index_document_indexes_text_and_image_metadata(tmp_path):
    pdf_path = tmp_path / "sample.txt"
    pdf_path.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    vectorstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma"))
    service = IndexingService(
        vectorstore=vectorstore,
        text_embedding_provider=MockTextEmbeddingProvider(),
        image_embedding_provider=MockImageEmbeddingProvider(),
        chunk_size=3,
        chunk_overlap=1,
    )

    stats = service.index_document(pdf_path, document_id="doc-index-1", storage_dir=tmp_path / "uploads")

    assert stats["status"] == "indexed"
    assert stats["document_id"] == "doc-index-1"
    assert stats["text_chunks"] >= 1
    assert vectorstore.document_exists("doc-index-1") is True

    text_hits = vectorstore.query_text(query_vector=[0.0, 0.1], n_results=5)
    assert text_hits
    assert text_hits[0]["document_id"] == "doc-index-1"


def test_index_document_detects_duplicates_by_sha256(tmp_path):
    source = tmp_path / "duplicate.txt"
    source.write_text("repeat once and again", encoding="utf-8")

    vectorstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma"))
    service = IndexingService(
        vectorstore=vectorstore,
        text_embedding_provider=MockTextEmbeddingProvider(),
        image_embedding_provider=MockImageEmbeddingProvider(),
        chunk_size=4,
        chunk_overlap=1,
    )

    first = service.index_document(source, document_id="doc-dup", storage_dir=tmp_path / "uploads")
    second = service.index_document(source, document_id="doc-dup", storage_dir=tmp_path / "uploads")

    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"
    assert second["text_chunks"] == 0
    assert second["images"] == 0


def test_default_image_caption_provider_creates_descriptive_fallback(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), color="blue").save(image_path)

    caption = DefaultImageCaptionProvider(model_loader=lambda: None).caption_image(image_path)

    assert "32" in caption and "24" in caption
    assert "image" in caption.lower()
