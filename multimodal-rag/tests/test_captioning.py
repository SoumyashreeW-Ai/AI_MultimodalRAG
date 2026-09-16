from pathlib import Path

from app.indexing.service import IndexingService
from app.models.document_models import DocumentAsset, ImageAsset, TextChunk


class MockVectorStore:
    def __init__(self):
        self.added_images = []

    def add_text_chunks(self, records, embeddings):
        pass

    def add_images(self, records, embeddings):
        self.added_images.append((records, embeddings))

    def document_exists(self, document_id, sha256=None):
        return False


class MockTextEmbeddingProvider:
    def embed_texts(self, texts):
        return [[0.0]] * len(texts)


class MockImageEmbeddingProvider:
    def embed_images(self, paths):
        return [[0.1, 0.2] for _ in paths]


class MockCaptionProvider:
    def __init__(self, fail_for=None):
        self.fail_for = set(fail_for or [])

    def caption_image(self, image_path):
        path = str(image_path)
        if path in self.fail_for:
            raise RuntimeError("caption failed")
        return f"caption for {Path(path).name}"


def _make_asset(tmp_path, n_images=1):
    doc_id = "doc-x"
    filename = "f.pdf"
    source = str(tmp_path / "f.pdf")
    images = []
    for i in range(n_images):
        img_path = str(tmp_path / f"img{i}.png")
        Path(img_path).write_text("fakeimage")
        images.append(
            ImageAsset(
                document_id=doc_id,
                filename=filename,
                source_path=source,
                page=i + 1,
                image_id=f"img-{i}",
                image_path=img_path,
                content_type="image/png",
            )
        )
    return DocumentAsset(
        document_id=doc_id,
        filename=filename,
        source_path=source,
        file_type="pdf",
        sha256="deadbeef",
        text_chunks=[],
        images=images,
    )


def test_caption_attached_to_image_metadata(monkeypatch, tmp_path):
    asset = _make_asset(tmp_path, n_images=1)

    def _stub_parse(path, document_id=None, storage_dir=None):
        return asset

    monkeypatch.setattr("app.indexing.service.parse_document", _stub_parse)

    store = MockVectorStore()
    captioner = MockCaptionProvider()
    svc = IndexingService(
        vectorstore=store,
        text_embedding_provider=MockTextEmbeddingProvider(),
        image_embedding_provider=MockImageEmbeddingProvider(),
        caption_provider=captioner,
    )

    tmp_file = tmp_path / "dummy.pdf"
    tmp_file.write_text("ok")
    result = svc.index_document(str(tmp_file), document_id=asset.document_id, storage_dir=str(tmp_path))

    assert result["status"] == "indexed"
    assert len(store.added_images) == 1
    records, embeddings = store.added_images[0]
    assert records[0]["caption"] == f"caption for img0.png"
    assert records[0]["metadata"]["caption"] == f"caption for img0.png"


def test_caption_failure_does_not_block_indexing(monkeypatch, tmp_path):
    asset = _make_asset(tmp_path, n_images=2)

    def _stub_parse(path, document_id=None, storage_dir=None):
        return asset

    monkeypatch.setattr("app.indexing.service.parse_document", _stub_parse)

    # Fail caption on second image path
    fail_path = asset.images[1].image_path
    captioner = MockCaptionProvider(fail_for=[fail_path])

    store = MockVectorStore()
    svc = IndexingService(
        vectorstore=store,
        text_embedding_provider=MockTextEmbeddingProvider(),
        image_embedding_provider=MockImageEmbeddingProvider(),
        caption_provider=captioner,
    )

    tmp_file = tmp_path / "dummy2.pdf"
    tmp_file.write_text("ok")
    result = svc.index_document(str(tmp_file), document_id=asset.document_id, storage_dir=str(tmp_path))

    assert result["status"] == "indexed"
    assert len(store.added_images) == 1
    records, embeddings = store.added_images[0]
    # first image should have caption, second should be empty or None
    assert records[0]["metadata"]["caption"] == f"caption for img0.png"
    assert records[1]["metadata"]["caption"] in ("", None)
