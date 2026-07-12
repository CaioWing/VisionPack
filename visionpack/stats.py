from __future__ import annotations

from collections import Counter
from typing import Any

from visionpack.core.project import Project


def collect_stats(project: Project) -> dict[str, Any]:
    # Streamed single pass: accumulate into counters/running totals so memory is
    # bounded by the number of distinct resolutions/classes, not by the dataset
    # size (no per-image lists held in RAM).
    class_counts: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    total_assets = 0
    total_annotations = 0
    total_objects = 0
    images_without_annotations = 0
    empty_images = 0
    total_size_bytes = 0

    for asset, annotation in project.index.iter_assets_with_annotations():
        total_assets += 1
        total_size_bytes += asset.size_bytes
        resolutions[f"{asset.width}x{asset.height}"] += 1
        if annotation is None:
            images_without_annotations += 1
        else:
            total_annotations += 1
        if annotation is None or not annotation.objects:
            empty_images += 1
        else:
            total_objects += len(annotation.objects)
            class_counts.update(obj.class_id for obj in annotation.objects)

    return {
        "assets": total_assets,
        "annotations": total_annotations,
        "objects": total_objects,
        "classes": len(project.manifest.classes),
        "class_distribution": dict(sorted(class_counts.items())),
        "resolutions": dict(sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))),
        "images_without_annotations": images_without_annotations,
        "empty_images": empty_images,
        "avg_labels_per_image": round(total_objects / total_assets, 3) if total_assets else 0,
        "total_size_bytes": total_size_bytes,
    }


def split_breakdown(project: Project, split_id: str = "default") -> dict[str, Any] | None:
    """Per-set image/object counts and class distribution for a versioned split.

    This is what makes metrics comparable as a dataset grows: you can confirm
    each set keeps a similar class balance from one snapshot to the next.
    """
    split = next((item for item in project.index.splits() if item.id == split_id), None)
    if split is None:
        return None

    sets: dict[str, Any] = {}
    for set_name in sorted(split.sets):
        asset_ids = split.sets[set_name]
        class_counts: Counter[str] = Counter()
        objects = 0
        without_annotations = 0
        for asset_id in asset_ids:
            annotation = project.index.annotation_for_asset(asset_id)
            if annotation and annotation.objects:
                class_counts.update(obj.class_id for obj in annotation.objects)
                objects += len(annotation.objects)
            else:
                without_annotations += 1
        sets[set_name] = {
            "images": len(asset_ids),
            "objects": objects,
            "images_without_annotations": without_annotations,
            "class_distribution": dict(sorted(class_counts.items())),
        }
    return {"split_id": split.id, "strategy": split.strategy, "locked": split.locked, "sets": sets}
