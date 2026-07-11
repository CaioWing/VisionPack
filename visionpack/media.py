from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from visionpack.core.errors import FormatError

# Extensions VisionPack will pick up during import discovery. Pillow can decode
# many more, but an explicit allowlist keeps import behavior predictable.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

# EXIF orientation tag and the values that imply width/height are swapped when
# the image is displayed. Annotation coordinates (YOLO/COCO) are expressed in the
# displayed frame, so probed dimensions must account for this.
_ORIENTATION_TAG = 0x0112
_SWAPPED_ORIENTATIONS = {5, 6, 7, 8}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def image_info(path: Path) -> tuple[int, int, int | None, str]:
    """Probe an image on disk, reading only what the codec needs for the header."""
    try:
        with Image.open(path) as img:
            return _probe(img, path)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise FormatError(f"File is not a readable image: {path} ({exc})") from exc


def image_info_from_bytes(data: bytes, source: Path) -> tuple[int, int, int | None, str]:
    """Probe an image already held in memory.

    Used by import, where the bytes have just been read to compute the content
    hash, so dimensions and hash come from a single read of the file.

    ``DecompressionBombError`` (a header claiming absurd dimensions) subclasses
    neither ``OSError`` nor ``ValueError``, so it is caught explicitly — one
    hostile file must become a per-file ingest failure, not abort the batch.
    """
    try:
        with Image.open(BytesIO(data)) as img:
            return _probe(img, source)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise FormatError(f"File is not a readable image: {source} ({exc})") from exc


def _probe(img: Image.Image, source: Path) -> tuple[int, int, int | None, str]:
    width, height = img.size
    channels = len(img.getbands()) or None
    image_format = (img.format or source.suffix.lstrip(".")).lower()
    if _exif_swaps_dimensions(img):
        width, height = height, width
    return width, height, channels, image_format


def _exif_swaps_dimensions(img: Image.Image) -> bool:
    getexif = getattr(img, "getexif", None)
    if getexif is None:
        return False
    try:
        orientation = getexif().get(_ORIENTATION_TAG)
    except Exception:  # noqa: BLE001 - corrupt EXIF must not break probing
        return False
    return orientation in _SWAPPED_ORIENTATIONS
