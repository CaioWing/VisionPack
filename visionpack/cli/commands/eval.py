from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.cli.output import emit_json
from visionpack.core.project import Project
from visionpack.eval import evaluate
from visionpack.predictions import FORMATS, load_predictions


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Score model predictions against a split's labels")
    parser.add_argument("predictions", help="Predictions file (vp/COCO JSON) or directory (YOLO txt)")
    parser.add_argument("--format", choices=list(FORMATS), default="auto", help="Predictions format (default: auto-detect)")
    parser.add_argument("--split", default="default", help="Split to evaluate against (default: 'default')")
    parser.add_argument("--set", dest="set_name", default="test", help="Set within the split (default: test)")
    parser.add_argument("--all", action="store_true", help="Evaluate on every asset instead of a split set (not a benchmark!)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for precision/recall (default: 0.25)")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    predictions = load_predictions(project, Path(args.predictions), fmt=args.format)
    result = evaluate(
        project,
        predictions,
        split_id=None if args.all else args.split,
        set_name=args.set_name,
        conf_threshold=args.conf,
    )
    if args.json:
        emit_json("eval", result)
        return 0

    scope = result["scope"]
    where = "all assets" if scope["set"] == "all" else f"split {scope['split']!r} / {scope['set']}"
    print(f"Evaluating {result['task']} on {where}: {result['images']} images, {result['images_with_predictions']} with predictions")
    if scope.get("locked") is False:
        print("warning: split is not locked; lock it (`vp split lock`) so this number stays comparable.")
    if result["unmatched_predictions"]:
        print(f"warning: {result['unmatched_predictions']} prediction image reference(s) matched no asset.")
    if result.get("unknown_classes"):
        print(f"warning: predictions referenced unknown classes: {', '.join(result['unknown_classes'])}")

    metrics = result["metrics"]
    if result["task"] == "classification":
        print(f"accuracy: {_fmt(metrics['accuracy'])} ({metrics['labeled_images']} labeled images)")
        if metrics["images_without_prediction"]:
            print(f"images without prediction: {metrics['images_without_prediction']}")
        print("per class (precision / recall / f1):")
        for class_id, row in result["per_class"].items():
            print(f"  {class_id}: {_fmt(row['precision'])} / {_fmt(row['recall'])} / {_fmt(row['f1'])} (gt={row['gt']})")
    else:
        print(f"mAP@50: {_fmt(metrics['mAP50'])}   mAP@50-95: {_fmt(metrics['mAP50_95'])}")
        print(f"precision: {_fmt(metrics['precision'])}   recall: {_fmt(metrics['recall'])}   (conf >= {result['conf_threshold']})")
        print("per class (AP@50 / precision / recall):")
        for class_id, row in result["per_class"].items():
            print(
                f"  {class_id}: {_fmt(row['ap50'])} / {_fmt(row['precision'])} / {_fmt(row['recall'])}"
                f" (gt={row['gt']}, preds={row['predictions']})"
            )
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
