from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from visionpack.core.project import Project


def collect_stats(project: Project) -> dict[str, Any]:
    assets = project.index.assets()
    annotations = project.index.annotations()
    annotations_by_asset = {annotation.asset_id: annotation for annotation in annotations}
    class_counts: Counter[str] = Counter()
    labels_per_image: list[int] = []
    resolutions: Counter[str] = Counter()
    sizes: list[int] = []

    for asset in assets:
        annotation = annotations_by_asset.get(asset.id)
        object_count = len(annotation.objects) if annotation else 0
        labels_per_image.append(object_count)
        sizes.append(asset.size_bytes)
        resolutions[f"{asset.width}x{asset.height}"] += 1
        if annotation:
            class_counts.update(obj.class_id for obj in annotation.objects)

    return {
        "assets": len(assets),
        "annotations": len(annotations),
        "objects": sum(class_counts.values()),
        "classes": len(project.manifest.classes),
        "class_distribution": dict(sorted(class_counts.items())),
        "resolutions": dict(sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))),
        "images_without_annotations": sum(1 for asset in assets if asset.id not in annotations_by_asset),
        "empty_images": sum(1 for count in labels_per_image if count == 0),
        "avg_labels_per_image": round(mean(labels_per_image), 3) if labels_per_image else 0,
        "total_size_bytes": sum(sizes),
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
