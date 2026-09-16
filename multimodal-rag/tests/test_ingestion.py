import subprocess
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from app.ingestion.parsers import parse_document
from app.ingestion.image_utils import safe_store_image, validate_image_file


def test_parse_text_markdown_file(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Heading\n\nThis is Markdown content.", encoding="utf-8")

    asset = parse_document(source, document_id="doc-md-1", storage_dir=tmp_path / "storage")

    assert asset.document_id == "doc-md-1"
    assert asset.filename == "notes.md"
    assert len(asset.text_chunks) >= 1
    assert "Markdown content" in asset.text_chunks[0].content


def test_parse_pdf_with_embedded_image(tmp_path):
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    img_path = tmp_path / "embedded.png"
    Image.new("RGB", (40, 40), color="blue").save(img_path)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF ingestion works")
    page.insert_image(fitz.Rect(100, 100, 180, 180), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()

    asset = parse_document(pdf_path, document_id="doc-pdf-1", storage_dir=tmp_path / "storage")

    assert any("PDF ingestion works" in chunk.content for chunk in asset.text_chunks)
    assert len(asset.images) >= 1
    assert asset.images[0].image_path


def test_parse_docx_file(tmp_path):
    from docx import Document

    docx_path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("DOCX ingestion works.")
    document.save(docx_path)

    asset = parse_document(docx_path, document_id="doc-docx-1", storage_dir=tmp_path / "storage")

    assert any("DOCX ingestion works" in chunk.content for chunk in asset.text_chunks)


def test_parse_legacy_doc_file(monkeypatch, tmp_path):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"fake legacy word binary")

    def fake_run(command, capture_output, text, check):
        assert command[0] in {"antiword", "catdoc"}
        return SimpleNamespace(returncode=0, stdout="Legacy doc content works.")

    monkeypatch.setattr(subprocess, "run", fake_run)

    asset = parse_document(source, document_id="doc-legacy-1", storage_dir=tmp_path / "storage")

    assert asset.file_type == "doc"
    assert any("Legacy doc content works" in chunk.content for chunk in asset.text_chunks)


def test_validate_and_store_image(tmp_path):
    source = tmp_path / "raw.png"
    Image.new("RGB", (30, 30), color="red").save(source)

    assert validate_image_file(source) is True

    stored_path = safe_store_image(source, target_dir=tmp_path / "safe_store", document_id="doc-image-1")

    assert stored_path.exists()
    assert stored_path.parent.name == "safe_store"
    assert stored_path.name.endswith((".png", ".jpg", ".jpeg"))


def test_image_uses_standard_jpeg_mime_type_and_content_based_id(tmp_path):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (30, 30), color="red").save(source)

    asset = parse_document(source, storage_dir=tmp_path / "storage")

    assert asset.images[0].content_type == "image/jpeg"
    first_id = asset.document_id

    # A replacement upload at the same path must get new Chroma record IDs.
    Image.new("RGB", (30, 30), color="blue").save(source)
    replacement = parse_document(source, storage_dir=tmp_path / "storage")
    assert replacement.document_id != first_id


def test_invalid_image_is_rejected(tmp_path):
    source = tmp_path / "not-an-image.png"
    source.write_text("not image data", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="Invalid image file"):
        parse_document(source, storage_dir=tmp_path / "storage")
