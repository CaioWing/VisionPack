from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("import", help="Import a dataset")
    parser.add_argument("source", help="Input dataset path")
    parser.add_argument("--format", required=True, choices=["yolo"], help="Input format")
    parser.add_argument("--task", default=None, choices=["detection"], help="Override project task")
    parser.add_argument("--copy", default="ingest", choices=["copy", "move", "hardlink", "reference", "ingest"], help="Asset copy mode")
    parser.add_argument("--class-map", help="Reserved for explicit class mapping files")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    if args.task and project.manifest.task != args.task:
        project.manifest.task = args.task
        project.save_manifest()
    summary = YoloImporter(project, Path(args.source), copy_mode=args.copy).run()
    print(
        "Imported YOLO dataset: "
        f"{summary.assets} assets, {summary.annotations} annotations, {summary.objects} objects"
    )
    if summary.orphan_labels:
        print(f"Warnings: {summary.orphan_labels} label files had no matching image")
    if summary.classes_added:
        print("Classes were discovered and written to visionpack.yaml")
    return 0
