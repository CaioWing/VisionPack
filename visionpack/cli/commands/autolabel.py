from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.autolabel import apply_predictions
from visionpack.cli.output import emit_json
from visionpack.core.project import Project
from visionpack.predictions import FORMATS, load_predictions


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("autolabel", help="Persist confident model predictions as annotations")
    parser.add_argument("predictions", help="Predictions file (vp/COCO JSON) or directory (YOLO txt)")
    parser.add_argument("--format", choices=list(FORMATS), default="auto", help="Predictions format (default: auto-detect)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Keep only objects at or above this confidence (default: 0.5)")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Also overwrite assets that already have annotations (default: only unlabeled assets are touched)",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    predictions = load_predictions(project, Path(args.predictions), fmt=args.format)
    summary = apply_predictions(project, predictions, min_confidence=args.min_confidence, replace=args.replace)
    if args.json:
        emit_json(
            "autolabel",
            {**summary, "min_confidence": args.min_confidence, "unknown_classes": sorted(set(predictions.unknown_classes))},
        )
        return 0
    print(f"Autolabeled {summary['labeled']} asset(s) with {summary['objects']} object(s) (source recorded as type=model).")
    if summary["skipped_existing"]:
        print(f"Skipped {summary['skipped_existing']} already-labeled asset(s); pass --replace to overwrite them.")
    if summary["skipped_low_confidence"]:
        print(f"Skipped {summary['skipped_low_confidence']} asset(s) whose predictions were all below {args.min_confidence}.")
    if summary["unmatched"]:
        print(f"warning: {summary['unmatched']} prediction image reference(s) matched no asset.")
    if predictions.unknown_classes:
        print(f"warning: predictions referenced unknown classes: {', '.join(sorted(set(predictions.unknown_classes)))}")
    return 0
