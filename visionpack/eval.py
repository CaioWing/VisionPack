"""Score model predictions against the dataset's own labels (``vp eval``).

A locked split + a snapshot already make a VisionPack dataset reproducible;
``vp eval`` turns it into a *benchmark*: predictions are scored against a
chosen set of a split (the test set by default), so numbers are comparable
across models and across time — and can't be inflated by evaluating on
training images.

Detection (and segmentation/keypoints, via each geometry's enclosing box) gets
COCO-style average precision: greedy confidence-ordered matching, 101-point
interpolated AP per class, reported at IoU 0.5 and averaged over 0.5:0.95.
Classification gets accuracy, per-class precision/recall/F1, and a confusion
matrix. Everything is returned as one JSON-friendly dict.
"""

from __future__ import annotations

from collections import defaultdict

from visionpack.core.errors import VisionPackError
from visionpack.core.models import BBox, ObjectAnnotation
from visionpack.core.project import Project
from visionpack.predictions import PredictionSet
from visionpack.split import get_split

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


def evaluate(
    project: Project,
    predictions: PredictionSet,
    *,
    split_id: str | None = "default",
    set_name: str = "test",
    conf_threshold: float = 0.25,
) -> dict:
    """Evaluate ``predictions`` on one set of a split (or the whole dataset).

    ``split_id=None`` evaluates on every asset — useful for sanity checks, but
    the result is not a benchmark number (it mixes training images in).
    """
    asset_ids, scope = _scope(project, split_id, set_name)
    result: dict = {
        "task": project.manifest.task,
        "scope": scope,
        "images": len(asset_ids),
        "images_with_predictions": sum(1 for asset_id in asset_ids if predictions.by_asset.get(asset_id)),
        "unmatched_predictions": len(predictions.unmatched),
    }
    if predictions.unknown_classes:
        result["unknown_classes"] = sorted(set(predictions.unknown_classes))

    if project.manifest.task == "classification":
        result.update(_evaluate_classification(project, predictions, asset_ids))
    else:
        result.update(_evaluate_detection(project, predictions, asset_ids, conf_threshold))
    return result


def _scope(project: Project, split_id: str | None, set_name: str) -> tuple[list[str], dict]:
    if split_id is None:
        return [asset.id for asset in project.index.assets()], {"split": None, "set": "all"}
    split = get_split(project, split_id)
    if split is None:
        raise VisionPackError(f"No split named {split_id!r}. Create one with `vp split create` (or pass --all).")
    if set_name not in split.sets:
        available = ", ".join(sorted(split.sets))
        raise VisionPackError(f"Split {split_id!r} has no set {set_name!r}. Available: {available}.")
    return list(split.sets[set_name]), {"split": split_id, "set": set_name, "locked": split.locked}


# --- detection ---------------------------------------------------------------


def _evaluate_detection(project: Project, predictions: PredictionSet, asset_ids: list[str], conf_threshold: float) -> dict:
    scope = set(asset_ids)
    truths: dict[str, dict[str, list[BBox]]] = defaultdict(lambda: defaultdict(list))  # class -> asset -> boxes
    gt_counts: dict[str, int] = defaultdict(int)
    for asset_id in asset_ids:
        annotation = project.index.annotation_for_asset(asset_id)
        if annotation is None:
            continue
        for obj in annotation.objects:
            if obj.bbox is None:
                continue
            truths[obj.class_id][asset_id].append(obj.bbox)
            gt_counts[obj.class_id] += 1

    preds: dict[str, list[tuple[float, str, BBox]]] = defaultdict(list)  # class -> (conf, asset, box)
    for asset_id, objects in predictions.by_asset.items():
        if asset_id not in scope:
            continue
        for obj in objects:
            if obj.bbox is None:
                continue
            preds[obj.class_id].append((obj.confidence if obj.confidence is not None else 1.0, asset_id, obj.bbox))

    class_ids = sorted(set(gt_counts) | set(preds))
    per_class: dict[str, dict] = {}
    micro_tp = micro_fp = micro_gt = 0
    for class_id in class_ids:
        class_preds = sorted(preds.get(class_id, []), key=lambda item: -item[0])
        num_gt = gt_counts.get(class_id, 0)
        # Each prediction's IoUs against its image's ground truth are computed
        # once and replayed for every threshold, instead of once per threshold.
        pred_ious = _pred_ious(class_preds, truths[class_id])
        aps = {threshold: _average_precision(_match(class_preds, pred_ious, threshold), num_gt) for threshold in IOU_THRESHOLDS}
        confident_indices = [index for index, item in enumerate(class_preds) if item[0] >= conf_threshold]
        confident = [class_preds[index] for index in confident_indices]
        tp = sum(_match(confident, [pred_ious[index] for index in confident_indices], 0.5))
        per_class[class_id] = {
            "gt": num_gt,
            "predictions": len(class_preds),
            "ap50": _round(aps[0.5]),
            "ap50_95": _round(_mean([value for value in aps.values() if value is not None])),
            "precision": _round(tp / len(confident)) if confident else None,
            "recall": _round(tp / num_gt) if num_gt else None,
        }
        micro_tp += tp
        micro_fp += len(confident) - tp
        micro_gt += num_gt

    scored = [item for item in per_class.values() if item["gt"] > 0]
    return {
        "conf_threshold": conf_threshold,
        "metrics": {
            "mAP50": _round(_mean([item["ap50"] for item in scored])),
            "mAP50_95": _round(_mean([item["ap50_95"] for item in scored])),
            "precision": _round(micro_tp / (micro_tp + micro_fp)) if micro_tp + micro_fp else None,
            "recall": _round(micro_tp / micro_gt) if micro_gt else None,
        },
        "per_class": per_class,
    }


def _pred_ious(class_preds: list[tuple[float, str, BBox]], gt_by_asset: dict[str, list[BBox]]) -> list[list[tuple[float, int]]]:
    """For each prediction, its ``(iou, gt_index)`` candidates, best IoU first.

    The sort is stable, so ties keep ground-truth enumeration order — matching
    then behaves exactly like a per-threshold argmax with ``>`` comparison.
    """
    ious: list[list[tuple[float, int]]] = []
    for _, asset_id, box in class_preds:
        candidates = [(bbox_iou(box, gt_box), index) for index, gt_box in enumerate(gt_by_asset.get(asset_id, []))]
        candidates = [item for item in candidates if item[0] > 0.0]
        candidates.sort(key=lambda item: -item[0])
        ious.append(candidates)
    return ious


def _match(class_preds: list[tuple[float, str, BBox]], pred_ious: list[list[tuple[float, int]]], iou_threshold: float) -> list[bool]:
    """Greedy COCO-style matching: each prediction (confidence-ordered) claims the
    best still-unmatched ground-truth box in its image. Returns TP flags."""
    matched: dict[str, set[int]] = defaultdict(set)
    flags: list[bool] = []
    for (_, asset_id, _), candidates in zip(class_preds, pred_ious, strict=True):
        hit = False
        for iou, gt_index in candidates:
            if iou < iou_threshold:
                break  # sorted best-first: nothing below qualifies either
            if gt_index in matched[asset_id]:
                continue
            matched[asset_id].add(gt_index)
            hit = True
            break
        flags.append(hit)
    return flags


def _average_precision(tp_flags: list[bool], num_gt: int) -> float | None:
    """101-point interpolated AP (the COCO convention) from confidence-ordered TP flags."""
    if num_gt == 0:
        return None
    if not tp_flags:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    tp = 0
    for index, flag in enumerate(tp_flags, start=1):
        tp += int(flag)
        precisions.append(tp / index)
        recalls.append(tp / num_gt)
    # Make precision monotonically decreasing, then sample 101 recall points.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    ap, pointer = 0.0, 0
    for step in range(101):
        recall_point = step / 100
        while pointer < len(recalls) and recalls[pointer] < recall_point:
            pointer += 1
        ap += precisions[pointer] if pointer < len(precisions) else 0.0
    return ap / 101


def bbox_iou(a: BBox, b: BBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


# --- classification ----------------------------------------------------------


def _evaluate_classification(project: Project, predictions: PredictionSet, asset_ids: list[str]) -> dict:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labeled = correct = without_prediction = 0
    for asset_id in asset_ids:
        annotation = project.index.annotation_for_asset(asset_id)
        if annotation is None or not annotation.objects:
            continue
        labeled += 1
        truth = annotation.objects[0].class_id
        predicted = _top_class(predictions.by_asset.get(asset_id, []))
        if predicted is None:
            without_prediction += 1
            confusion[truth]["(none)"] += 1
            continue
        confusion[truth][predicted] += 1
        if predicted == truth:
            correct += 1

    per_class: dict[str, dict] = {}
    class_ids = sorted({item.id for item in project.manifest.classes} | set(confusion))
    for class_id in class_ids:
        gt = sum(confusion[class_id].values())
        tp = confusion[class_id].get(class_id, 0)
        predicted_as = sum(row.get(class_id, 0) for row in confusion.values())
        per_class[class_id] = {
            "gt": gt,
            "precision": _round(tp / predicted_as) if predicted_as else None,
            "recall": _round(tp / gt) if gt else None,
            "f1": _round(2 * tp / (predicted_as + gt)) if predicted_as + gt else None,
        }

    return {
        "metrics": {
            "accuracy": _round(correct / labeled) if labeled else None,
            "labeled_images": labeled,
            "images_without_prediction": without_prediction,
        },
        "per_class": per_class,
        "confusion_matrix": {truth: dict(row) for truth, row in confusion.items()},
    }


def _top_class(objects: list[ObjectAnnotation]) -> str | None:
    if not objects:
        return None
    return max(objects, key=lambda obj: obj.confidence if obj.confidence is not None else 1.0).class_id


# --- shared ------------------------------------------------------------------


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
