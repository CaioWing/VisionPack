"""Active-learning queue (``vp queue``): rank images by annotation value.

In a vertical dataset the scarce resource is human labeling time, so the queue
answers "which images should a person look at next?". Given optional model
predictions it ranks:

- **unlabeled images with no predictions** highest (score 1.0) — the model has
  nothing to say, a human must;
- **unlabeled images by model uncertainty** — ``1 - mean confidence``; the
  confident ones are ``vp autolabel`` material, the uncertain ones are where a
  human label teaches the model the most;
- **labeled images by ground-truth/prediction disagreement** (with
  ``include_labeled``) — a confident prediction with no matching label suggests
  a *missing* label; a label the model never finds suggests a *wrong or stale*
  one. Both are label-quality audit signals, not just curation.
"""

from __future__ import annotations

from typing import Any

from visionpack.core.project import Project
from visionpack.eval import bbox_iou
from visionpack.predictions import PredictionSet


def rank_for_annotation(
    project: Project,
    predictions: PredictionSet | None = None,
    *,
    include_labeled: bool = False,
    confident: float = 0.5,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for asset in project.index.assets():
        annotation = project.index.annotation_for_asset(asset.id)
        predicted = predictions.by_asset.get(asset.id, []) if predictions else []
        has_labels = annotation is not None and bool(annotation.objects)

        if not has_labels:
            if not predicted:
                score, reasons = 1.0, ["unlabeled, no model predictions"]
            else:
                confidences = [obj.confidence if obj.confidence is not None else 1.0 for obj in predicted]
                mean = sum(confidences) / len(confidences)
                score = 1.0 - mean
                reasons = [f"unlabeled, model uncertain (mean confidence {mean:.2f}, {len(predicted)} objects)"]
        elif include_labeled and predictions is not None:
            score, reasons = _disagreement(annotation.objects, predicted, confident, iou_threshold)
            if score == 0.0:
                continue
        else:
            continue
        items.append({"asset_id": asset.id, "path": asset.original_path, "score": round(score, 4), "reasons": reasons})

    items.sort(key=lambda item: (-item["score"], item["asset_id"]))
    return items


def _disagreement(gt_objects: list, predicted: list, confident: float, iou_threshold: float) -> tuple[float, list[str]]:
    """Fraction of objects on which the model and the labels disagree."""
    gt_boxes = [(obj.class_id, obj.bbox) for obj in gt_objects if obj.bbox is not None]
    confident_preds = [obj for obj in predicted if (obj.confidence or 0.0) >= confident and obj.bbox is not None]

    matched_gt: set[int] = set()
    unmatched_preds = 0
    for pred in sorted(confident_preds, key=lambda obj: -(obj.confidence or 0.0)):
        best_index = -1
        best_iou = 0.0
        for index, (class_id, box) in enumerate(gt_boxes):
            if index in matched_gt or class_id != pred.class_id:
                continue
            iou = bbox_iou(pred.bbox, box)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_index)
        else:
            unmatched_preds += 1
    unmatched_gt = len(gt_boxes) - len(matched_gt)

    total = len(gt_boxes) + len(confident_preds)
    if total == 0:
        return 0.0, []
    reasons: list[str] = []
    if unmatched_preds:
        reasons.append(f"possible missing labels: {unmatched_preds} confident prediction(s) match no ground truth")
    if unmatched_gt:
        reasons.append(f"possible wrong/stale labels: {unmatched_gt} ground-truth object(s) the model never finds")
    return min((unmatched_preds + unmatched_gt) / total, 1.0), reasons
