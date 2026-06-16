from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.project import Project
from visionpack.formats.classification import export_imagefolder
from visionpack.formats.coco import export_coco
from visionpack.formats.yolo import export_yolo
from visionpack.progress import cli_progress
from visionpack.snapshot import open_snapshot


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export a dataset")
    parser.add_argument("--format", required=True, choices=["yolo", "coco", "imagefolder"], help="Output format")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--split",
        nargs="?",
        const="default",
        default=None,
        help="Export into train/val/test using a split (defaults to 'default' when given without a value)",
    )
    parser.add_argument(
        "--snapshot",
        help="Export the dataset as it was at this snapshot version (e.g. v2) instead of the current state",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    if args.snapshot:
        project = open_snapshot(project, args.snapshot)
    output = Path(args.output)
    with cli_progress(f"Exporting {args.format}") as callback:
        if args.format == "coco":
            summary = export_coco(project, output, split_id=args.split, progress=callback)
            detail = f"{summary['images']} images, {summary['annotations']} annotations, {summary['objects']} objects"
            message = f"Exported COCO dataset to {output.resolve()}: {detail}"
        elif args.format == "imagefolder":
            summary = export_imagefolder(project, output, split_id=args.split, progress=callback)
            message = f"Exported ImageFolder dataset to {output.resolve()}: {summary['images']} images"
        else:
            summary = export_yolo(project, output, split_id=args.split, progress=callback)
            detail = f"{summary['images']} images, {summary['labels']} label files, {summary['objects']} objects"
            message = f"Exported YOLO dataset to {output.resolve()}: {detail}"
    print(message)

    streamed = summary.get("streamed")
    if streamed:
        print(
            f"{streamed} cloud-backed image(s) were not copied; their URIs were written to "
            f"{(output / 'manifest.jsonl').resolve()} for streaming."
        )

    if args.split:
        sets = ", ".join(f"{name}={count}" for name, count in summary.get("sets", {}).items())
        print(f"Split {args.split!r}: {sets}")
        if summary.get("skipped"):
            print(f"Skipped {summary['skipped']} assets not assigned to any set in split {args.split!r}")
    return 0
