from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from visionpack.core.errors import FormatError

# dHash ("difference hash"): downscale to grayscale and encode whether each pixel
# is brighter than its right neighbor. 64 bits, robust to re-encoding, mild
# scaling and compression, while staying dependency-free (just Pillow, which we
# already use). Two images are "near duplicates" when their hashes are within a
# small Hamming distance.
_HASH_SIZE = 8
_BITS = _HASH_SIZE * _HASH_SIZE  # 64
_RESAMPLE = Image.Resampling.BILINEAR


def dhash_bytes(data: bytes) -> str:
    """Perceptual hash of an in-memory image (used at import, reusing read bytes)."""
    try:
        with Image.open(BytesIO(data)) as img:
            return _dhash(img)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise FormatError(f"Cannot compute perceptual hash: {exc}") from exc


def dhash_path(path: Path) -> str:
    """Perceptual hash of an image on disk (used to backfill older datasets)."""
    try:
        with Image.open(path) as img:
            return _dhash(img)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise FormatError(f"Cannot compute perceptual hash for {path}: {exc}") from exc


def _dhash(img: Image.Image) -> str:
    # +1 column so we can compare each pixel against its right neighbor.
    small = img.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), _RESAMPLE)
    # "L" mode: tobytes() is one byte per pixel, row-major (no getdata deprecation).
    pixels = small.tobytes()
    stride = _HASH_SIZE + 1
    bits = 0
    for row in range(_HASH_SIZE):
        base = row * stride
        for col in range(_HASH_SIZE):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    """Number of differing bits between two hex perceptual hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def band_keys(value: int, bands: int) -> list[tuple[int, int]]:
    """Split a 64-bit hash into ``bands`` contiguous chunks for LSH bucketing.

    Pigeonhole: two hashes within Hamming distance ``d`` differ in at most ``d``
    bits, so with ``bands = d + 1`` at least one band is identical and the pair
    lands in a shared bucket. That makes candidate generation exact (no missed
    pairs) while avoiding an all-pairs O(n^2) scan on large datasets.
    """
    width = (_BITS + bands - 1) // bands
    mask = (1 << width) - 1
    return [(i, (value >> (i * width)) & mask) for i in range(bands)]
