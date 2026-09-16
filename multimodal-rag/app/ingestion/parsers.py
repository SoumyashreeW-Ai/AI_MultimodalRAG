from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from PIL import Image

from app.models.document_models import DocumentAsset, ImageAsset, TextChunk

from .image_utils import compute_sha256, safe_store_image, validate_image_file
from .audio_utils import transcribe_audio
import shutil

SUPPORTED_FILE_TYPES = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
}


def _default_document_id(path: Path) -> str:
    # Use the file contents instead of its upload path. Uploads commonly reuse a
    # filename, and path-derived IDs would otherwise reuse Chroma record IDs for
    # a changed PDF or image.
    return compute_sha256(path)[:16]


def _image_content_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _chunk_text(document_id: str, filename: str, source_path: str, page: int, content: str) -> TextChunk:
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("Cannot create a chunk from empty content")
    chunk_id = f"{document_id}:page-{page}:chunk-1"
    return TextChunk(
        document_id=document_id,
        filename=filename,
        source_path=source_path,
        page=page,
        chunk_id=chunk_id,
        content=cleaned,
        content_type="text",
    )


def _handle_image_file(document_id: str, file_path: Path, storage_dir: Path, page: int = 1) -> ImageAsset:
    if not validate_image_file(file_path):
        raise ValueError(f"Invalid image file: {file_path}")

    stored_path = safe_store_image(file_path, target_dir=storage_dir, document_id=document_id)
    with Image.open(file_path) as image:
        width, height = image.size

    image_id = f"{document_id}:page-{page}:image-1"
    return ImageAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        page=page,
        image_id=image_id,
        image_path=str(stored_path),
        content_type=_image_content_type(file_path),
        image_width=width,
        image_height=height,
    )


def _parse_pdf(document_id: str, file_path: Path, storage_dir: Path) -> DocumentAsset:
    import fitz

    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type="pdf",
        sha256=compute_sha256(file_path),
    )

    with fitz.open(file_path) as pdf_document:
        for page_number, page in enumerate(pdf_document, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                asset.text_chunks.append(
                    _chunk_text(document_id, file_path.name, str(file_path), page_number, text)
                )

            for image_number, image in enumerate(page.get_images(full=True), start=1):
                try:
                    xref = image[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    # PyMuPDF can return CMYK embedded images. Convert those to
                    # RGB before PNG encoding so Pillow can validate and index
                    # them reliably. Grayscale and alpha PNGs are already safe.
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_temp = storage_dir / f"{document_id}-page-{page_number}-img-{image_number}.png"
                    image_temp.parent.mkdir(parents=True, exist_ok=True)
                    image_temp.write_bytes(pix.tobytes("png"))
                    stored_path = safe_store_image(
                        image_temp,
                        target_dir=storage_dir,
                        document_id=f"{document_id}:page-{page_number}:image-{image_number}",
                    )
                    asset.images.append(
                        ImageAsset(
                            document_id=document_id,
                            filename=file_path.name,
                            source_path=str(file_path),
                            page=page_number,
                            image_id=f"{document_id}:page-{page_number}:image-{image_number}",
                            image_path=str(stored_path),
                            content_type="image/png",
                            image_width=pix.width,
                            image_height=pix.height,
                        )
                    )
                except Exception:
                    continue
    return asset


def _extract_legacy_doc_text(file_path: Path) -> str:
    """Best-effort extraction for legacy .doc files using system converters.

    This keeps the fix minimal and avoids changing the existing audio fallback.
    """
    for command in (["antiword", str(file_path)], ["catdoc", str(file_path)]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    raise RuntimeError(
        "Legacy .doc files require antiword or catdoc to extract text. "
        "Install one of those converters and retry the upload."
    )


def _parse_legacy_doc(document_id: str, file_path: Path) -> DocumentAsset:
    content = _extract_legacy_doc_text(file_path)
    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type="doc",
        sha256=compute_sha256(file_path),
    )
    if content.strip():
        asset.text_chunks.append(
            _chunk_text(document_id, file_path.name, str(file_path), page=1, content=content)
        )
    return asset


def _parse_docx(document_id: str, file_path: Path, storage_dir: Path) -> DocumentAsset:
    document = DocxDocument(str(file_path))
    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type="docx",
        sha256=compute_sha256(file_path),
    )

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    combined = "\n".join(paragraphs)
    if combined:
        asset.text_chunks.append(
            _chunk_text(document_id, file_path.name, str(file_path), page=1, content=combined)
        )

    for image_number, shape in enumerate(document.inline_shapes, start=1):
        try:
            inline = getattr(shape, "_inline", None)
            blip = getattr(inline, "blip", None)
            r_id = getattr(blip, "rId", None)
            if r_id is None:
                continue
            related = document.part.rels[r_id]
            image_part = related.target_part
            blob = image_part.blob
            ext = ".png"
            content_type = image_part.content_type or "image/png"
            if "/" in content_type:
                ext = f".{content_type.split('/')[-1]}"
            image_temp = storage_dir / f"{document_id}-docx-image-{image_number}{ext}"
            image_temp.parent.mkdir(parents=True, exist_ok=True)
            image_temp.write_bytes(blob)
            if validate_image_file(image_temp):
                stored_path = safe_store_image(
                    image_temp,
                    target_dir=storage_dir,
                    document_id=f"{document_id}:page-1:image-{image_number}",
                )
                with Image.open(image_temp) as image:
                    width, height = image.size
                asset.images.append(
                    ImageAsset(
                        document_id=document_id,
                        filename=file_path.name,
                        source_path=str(file_path),
                        page=1,
                        image_id=f"{document_id}:page-1:image-{image_number}",
                        image_path=str(stored_path),
                        content_type=content_type,
                        image_width=width,
                        image_height=height,
                    )
                )
        except Exception:
            continue

    return asset


def _parse_text_file(document_id: str, file_path: Path) -> DocumentAsset:
    content = file_path.read_text(encoding="utf-8", errors="replace")
    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type=file_path.suffix.lower().lstrip("."),
        sha256=compute_sha256(file_path),
    )
    if content.strip():
        asset.text_chunks.append(
            _chunk_text(document_id, file_path.name, str(file_path), page=1, content=content)
        )
    return asset


def _parse_image_file(document_id: str, file_path: Path, storage_dir: Path) -> DocumentAsset:
    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type="image",
        sha256=compute_sha256(file_path),
    )
    if not validate_image_file(file_path):
        raise ValueError(f"Invalid image file: {file_path}")
    asset.images.append(
        _handle_image_file(document_id, file_path, storage_dir=storage_dir, page=1)
    )
    return asset


def _parse_audio_file(document_id: str, file_path: Path, storage_dir: Path) -> DocumentAsset:
    asset = DocumentAsset(
        document_id=document_id,
        filename=file_path.name,
        source_path=str(file_path),
        file_type="audio",
        sha256=compute_sha256(file_path),
    )

    # copy audio into storage_dir for stable serving
    target = storage_dir / f"{document_id}-{file_path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(file_path, target)
    except Exception:
        target = file_path

    try:
        segments = transcribe_audio(target)
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc

    for idx, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        chunk_id = f"{document_id}:segment-{idx}"
        chunk = TextChunk(
            document_id=document_id,
            filename=file_path.name,
            source_path=str(target),
            page=idx,
            chunk_id=chunk_id,
            content=text,
            content_type="audio",
            start_time=float(seg.get("start", 0.0)),
            end_time=float(seg.get("end", 0.0)),
        )
        asset.text_chunks.append(chunk)

    return asset


def parse_document(
    source_path: str | Path,
    document_id: str | None = None,
    storage_dir: str | Path = "data/uploads",
) -> DocumentAsset:
    file_path = Path(source_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document does not exist: {file_path}")

    resolved_document_id = document_id or _default_document_id(file_path)
    target_dir = Path(storage_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(resolved_document_id, file_path, target_dir)
    if suffix == ".doc":
        return _parse_legacy_doc(resolved_document_id, file_path)
    if suffix == ".docx":
        return _parse_docx(resolved_document_id, file_path, target_dir)
    if suffix in {".txt", ".md", ".markdown"}:
        return _parse_text_file(resolved_document_id, file_path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _parse_image_file(resolved_document_id, file_path, target_dir)
    if suffix in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return _parse_audio_file(resolved_document_id, file_path, target_dir)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")
