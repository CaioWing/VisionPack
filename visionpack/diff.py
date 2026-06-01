from __future__ import annotations

from typing import Any

from visionpack.core.project import Project
from visionpack.snapshot import load_snapshot


def diff_snapshots(project: Project, left: str, right: str) -> dict[str, Any]:
    old = load_snapshot(project, left)
    new = load_snapshot(project, right)
    old_inv = old["inventory"]
    new_inv = new["inventory"]

    old_assets = set(old_inv["assets"])
    new_assets = set(new_inv["assets"])
    old_annotations = old_inv["annotations"]
    new_annotations = new_inv["annotations"]
    modified_annotations = sorted(
        asset_id
        for asset_id in set(old_annotations) & set(new_annotations)
        if old_annotations[asset_id] != new_annotations[asset_id]
    )

    old_classes = {item["id"] for item in old_inv.get("classes", [])}
    new_classes = {item["id"] for item in new_inv.get("classes", [])}
    return {
        "from": left,
        "to": right,
        "assets_added": sorted(new_assets - old_assets),
        "assets_removed": sorted(old_assets - new_assets),
        "annotations_added": sorted(set(new_annotations) - set(old_annotations)),
        "annotations_removed": sorted(set(old_annotations) - set(new_annotations)),
        "annotations_modified": modified_annotations,
        "classes_added": sorted(new_classes - old_classes),
        "classes_removed": sorted(old_classes - new_classes),
        "splits_changed": old_inv.get("splits", {}) != new_inv.get("splits", {}),
        "stats_before": old.get("stats", {}),
        "stats_after": new.get("stats", {}),
    }
