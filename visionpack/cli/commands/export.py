from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.project import Project
from visionpack.formats.coco import export_coco
from visionpack.formats.yolo import export_yolo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export a dataset")
    parser.add_argument("--format", required=True, choices=["yolo", "coco"], help="Output format")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--split",
        nargs="?",
        const="default",
        default=None,
        help="Export into train/val/test using a split (defaults to 'default' when given without a value)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    output = Path(args.output)
    if args.format == "coco":
        summary = export_coco(project, output, split_id=args.split)
        detail = f"{summary['images']} images, {summary['annotations']} annotations, {summary['objects']} objects"
        print(f"Exported COCO dataset to {output.resolve()}: {detail}")
    else:
        summary = export_yolo(project, output, split_id=args.split)
        detail = f"{summary['images']} images, {summary['labels']} label files, {summary['objects']} objects"
        print(f"Exported YOLO dataset to {output.resolve()}: {detail}")

    if args.split:
        sets = ", ".join(f"{name}={count}" for name, count in summary.get("sets", {}).items())
        print(f"Split {args.split!r}: {sets}")
        if summary.get("skipped"):
            print(f"Skipped {summary['skipped']} assets not assigned to any set in split {args.split!r}")
    return 0
