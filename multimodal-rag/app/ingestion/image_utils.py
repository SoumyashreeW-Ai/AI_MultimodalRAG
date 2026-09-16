from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def compute_sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_file(path: str | Path) -> bool:
    file_path = Path(path)
    if not file_path.exists():
        return False
    if file_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return False
    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def safe_store_image(
    source_path: str | Path,
    target_dir: str | Path,
    document_id: str,
) -> Path:
    source = Path(source_path)
    target_root = Path(target_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    if not validate_image_file(source):
        raise ValueError(f"Invalid image file: {source}")

    suffix = source.suffix.lower() if source.suffix else ".png"
    digest = compute_sha256(source)
    safe_name = f"{document_id}-{digest[:12]}{suffix}"
    destination = (target_root / safe_name).resolve()
    target_root_resolved = target_root.resolve()
    if target_root_resolved not in destination.parents and destination != target_root_resolved:
        raise ValueError("Refusing to store image outside of target directory")

    destination.write_bytes(source.read_bytes())
    return destination
