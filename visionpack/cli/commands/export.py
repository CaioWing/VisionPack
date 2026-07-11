from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.cli.output import emit_json
from visionpack.core.project import Project
from visionpack.formats.classification import export_imagefolder
from visionpack.formats.coco import export_coco
from visionpack.formats.masks import export_masks
from visionpack.formats.yolo import export_yolo
from visionpack.progress import cli_progress
from visionpack.snapshot import open_snapshot


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export a dataset")
    parser.add_argument("--format", required=True, choices=["yolo", "coco", "imagefolder", "masks"], help="Output format")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--seg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="YOLO only: write YOLO-seg polygon labels (defaults to on for segmentation projects)",
    )
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
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    if args.snapshot:
        project = open_snapshot(project, args.snapshot)
    output = Path(args.output)
    if args.json:
        return _run_json(args, project, output)
    with cli_progress(f"Exporting {args.format}") as callback:
        if args.format == "coco":
            summary = export_coco(project, output, split_id=args.split, progress=callback)
            detail = f"{summary['images']} images, {summary['annotations']} annotations, {summary['objects']} objects"
            message = f"Exported COCO dataset to {output.resolve()}: {detail}"
        elif args.format == "imagefolder":
            summary = export_imagefolder(project, output, split_id=args.split, progress=callback)
            message = f"Exported ImageFolder dataset to {output.resolve()}: {summary['images']} images"
        elif args.format == "masks":
            summary = export_masks(project, output, split_id=args.split, progress=callback)
            detail = f"{summary['images']} images, {summary['masks']} masks, {summary['objects']} objects"
            message = f"Exported semantic masks to {output.resolve()}: {detail}"
        else:
            summary = export_yolo(project, output, split_id=args.split, progress=callback, seg=args.seg)
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


def _run_json(args: argparse.Namespace, project: Project, output: Path) -> int:
    # No progress bar: stdout carries exactly one JSON document.
    if args.format == "coco":
        summary = export_coco(project, output, split_id=args.split)
    elif args.format == "imagefolder":
        summary = export_imagefolder(project, output, split_id=args.split)
    elif args.format == "masks":
        summary = export_masks(project, output, split_id=args.split)
    else:
        summary = export_yolo(project, output, split_id=args.split, seg=args.seg)
    emit_json(
        "export",
        {
            "format": args.format,
            "output": str(output.resolve()),
            "split": args.split,
            "snapshot": args.snapshot,
            **summary,
        },
    )
    return 0
