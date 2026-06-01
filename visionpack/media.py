from __future__ import annotations

import struct
from pathlib import Path

from visionpack.core.errors import FormatError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def image_info(path: Path) -> tuple[int, int, int | None, str]:
    with path.open("rb") as handle:
        header = handle.read(64)

    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width, height = struct.unpack(">II", header[16:24])
        return width, height, None, "png"
    if header[:2] == b"\xff\xd8":
        width, height = _jpeg_size(path)
        return width, height, 3, "jpeg"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        width, height = struct.unpack("<HH", header[6:10])
        return width, height, None, "gif"
    if header.startswith(b"BM") and len(header) >= 26:
        width, height = struct.unpack("<ii", header[18:26])
        return abs(width), abs(height), None, "bmp"
    raise FormatError(f"File is not a readable image: {path}")


def _jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        handle.read(2)
        while True:
            byte = handle.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                handle.read(3)
                height, width = struct.unpack(">HH", handle.read(4))
                return width, height
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            handle.seek(length - 2, 1)
    raise FormatError(f"Could not read JPEG dimensions: {path}")
