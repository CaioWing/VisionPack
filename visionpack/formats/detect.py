"""Import-format detection for ``vp import --format auto``.

Given the path a user points ``vp import`` at, decide whether it is a COCO
annotations file, a YOLO dataset root, or an ImageFolder root. Detection is by
*structure*, in order of decreasing certainty:

1. a ``.json`` file (or a directory whose root holds an instances-style JSON)
   is **COCO**;
2. any ``.txt`` label files, or YOLO furniture (``classes.txt``, ``obj.names``,
   ``data.yaml``), means **YOLO**;
3. otherwise, images living only under first-level subdirectories (the
   folder-per-class convention) means **ImageFolder**;
4. images sitting directly in the root (no labels at all) fall back to
   **YOLO**, which imports unlabeled images fine.

Anything that fits none of these raises a :class:`VisionPackError` telling the
user to pass ``--format`` explicitly — a wrong silent guess would import the
dataset with the wrong task, which costs far more than one flag.
"""

from __future__ import annotations

from pathlib import Path

from visionpack.core.errors import VisionPackError
from visionpack.media import is_image_path

_YOLO_FURNITURE = ("classes.txt", "obj.names", "data.yaml")


def detect_import_format(source: Path) -> str:
    """The import format (``yolo`` | ``coco`` | ``imagefolder``) of ``source``."""
    source = source.resolve()
    if not source.exists():
        raise VisionPackError(f"Import source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() == ".json":
            return "coco"
        raise VisionPackError(
            f"Cannot auto-detect the format of a single file: {source}. "
            "Pass --format explicitly (a COCO annotations file must end in .json)."
        )

    # COCO: an instances-style JSON at the root.
    if coco_json_in(source) is not None:
        return "coco"

    # YOLO: label files or the classes/data furniture anywhere relevant.
    if any((source / name).exists() for name in _YOLO_FURNITURE):
        return "yolo"
    has_txt = any(path.suffix.lower() == ".txt" for path in source.rglob("*.txt"))
    if has_txt:
        return "yolo"

    top_level_images = any(path.is_file() and is_image_path(path) for path in source.iterdir())
    class_dirs = [path for path in source.iterdir() if path.is_dir()]
    nested_images = any(
        any(child.is_file() and is_image_path(child) for child in class_dir.rglob("*")) for class_dir in class_dirs
    )
    if not top_level_images and nested_images:
        return "imagefolder"
    if top_level_images:
        return "yolo"  # plain images, no labels: YOLO import handles unlabeled sets

    raise VisionPackError(
        f"Cannot auto-detect the dataset format under {source} (no labels, no images found). "
        "Pass --format yolo|coco|imagefolder explicitly."
    )


def coco_json_in(source: Path) -> Path | None:
    """The first instances-style JSON directly under ``source``, if any."""
    for path in sorted(source.glob("*.json")):
        if _looks_like_coco(path):
            return path
    return None


def _looks_like_coco(path: Path) -> bool:
    """Cheap structural sniff: a JSON object mentioning images+annotations.

    Reads only the head of the file, so a multi-hundred-MB instances JSON
    doesn't get fully parsed just to answer "is this COCO?".
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return head.lstrip().startswith("{") and any(key in head for key in ('"images"', '"annotations"', '"categories"'))
