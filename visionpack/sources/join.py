from __future__ import annotations

from dataclasses import dataclass

from visionpack.core.errors import VisionPackError
from visionpack.sources.resolver import FileRef


@dataclass(slots=True)
class JoinResult:
    pairs: list[tuple[FileRef, FileRef | None]]  # (image, label or None)
    images_without_label: int
    labels_without_image: list[FileRef]


def join_refs(images: list[FileRef], labels: list[FileRef], match: str) -> JoinResult:
    """Pair each image with its label by the chosen key.

    - ``relpath``: same path (minus extension) under each root — the YOLO
      ``images/`` ÷ ``labels/`` parallel-tree convention, even across repos.
    - ``stem``: just the filename matches, ignoring directory structure — for
      datasets whose two sides are laid out differently but use unique names.
    """
    if match == "relpath":
        key = lambda ref: ref.relkey  # noqa: E731
    elif match == "stem":
        key = lambda ref: ref.stem  # noqa: E731
    else:
        raise VisionPackError(
            f"Unknown match strategy {match!r}. Use 'relpath', 'stem', or 'embedded' (COCO)."
        )

    label_by_key: dict[str, FileRef] = {}
    for label in labels:
        # First one wins; deterministic because resolvers list sorted.
        label_by_key.setdefault(key(label), label)

    pairs: list[tuple[FileRef, FileRef | None]] = []
    matched_keys: set[str] = set()
    images_without_label = 0
    for image in images:
        matched = label_by_key.get(key(image))
        pairs.append((image, matched))
        if matched is None:
            images_without_label += 1
        else:
            matched_keys.add(key(image))

    labels_without_image = [label for label in labels if key(label) not in matched_keys]
    return JoinResult(pairs, images_without_label, labels_without_image)
