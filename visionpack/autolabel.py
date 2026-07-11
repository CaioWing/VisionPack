"""Model-assisted labeling (``vp autolabel``): turn predictions into annotations.

This is the labeling half of the model-in-the-loop cycle: run any model over
the dataset (or an export of it), load its predictions, and persist the
confident ones as annotations with ``source.type = "model"`` — so model labels
are always distinguishable from human/import labels, auditable, and re-doable.

The default policy is conservative: only unlabeled assets are touched, and only
objects at or above ``min_confidence`` are kept. ``replace=True`` opts into
overwriting existing annotations (e.g. refreshing a previous autolabel pass
with a better model); it still never runs implicitly.
"""

from __future__ import annotations

from typing import Any

from visionpack.core.models import Annotation, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.predictions import PredictionSet


def apply_predictions(
    project: Project,
    predictions: PredictionSet,
    *,
    min_confidence: float = 0.5,
    replace: bool = False,
) -> dict[str, Any]:
    labeled = 0
    objects = 0
    skipped_existing = 0
    skipped_low_confidence = 0

    for asset_id in sorted(predictions.by_asset):
        kept = [obj for obj in predictions.by_asset[asset_id] if (obj.confidence or 0.0) >= min_confidence]
        if project.manifest.task == "classification" and kept:
            kept = [_top(kept)]
        if not kept:
            skipped_low_confidence += 1
            continue
        if project.index.annotation_for_asset(asset_id) is not None and not replace:
            skipped_existing += 1
            continue
        annotation = Annotation(
            id=f"ann_{asset_id}",
            asset_id=asset_id,
            task=project.manifest.task,
            format="internal",
            objects=kept,
            source={"type": "model", "origin": predictions.origin, "min_confidence": min_confidence, "labeled_at": utc_now()},
        )
        project.index.upsert_annotation(annotation)
        labeled += 1
        objects += len(kept)

    project.index.save()
    return {
        "labeled": labeled,
        "objects": objects,
        "skipped_existing": skipped_existing,
        "skipped_low_confidence": skipped_low_confidence,
        "unmatched": len(predictions.unmatched),
    }


def _top(objects: list[ObjectAnnotation]) -> ObjectAnnotation:
    return max(objects, key=lambda obj: obj.confidence if obj.confidence is not None else 1.0)
