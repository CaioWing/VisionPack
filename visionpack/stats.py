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
        "resolutions": dict(resolutions.most_common()),
        "images_without_annotations": sum(1 for asset in assets if asset.id not in annotations_by_asset),
        "empty_images": sum(1 for count in labels_per_image if count == 0),
        "avg_labels_per_image": round(mean(labels_per_image), 3) if labels_per_image else 0,
        "total_size_bytes": sum(sizes),
    }
