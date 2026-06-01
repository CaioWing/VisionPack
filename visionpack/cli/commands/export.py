from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.project import Project
from visionpack.formats.yolo import export_yolo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export a dataset")
    parser.add_argument("--format", required=True, choices=["yolo"], help="Output format")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    summary = export_yolo(project, Path(args.output))
    print(
        f"Exported YOLO dataset to {Path(args.output).resolve()}: "
        f"{summary['images']} images, {summary['labels']} label files, {summary['objects']} objects"
    )
    return 0
