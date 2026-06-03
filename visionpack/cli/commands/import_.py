from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project
from visionpack.formats.classification import ImageFolderImporter
from visionpack.formats.coco import CocoImporter
from visionpack.formats.yolo import YoloImporter


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("import", help="Import a dataset")
    parser.add_argument("source", help="Input dataset path (YOLO/ImageFolder root, or COCO annotation JSON)")
    parser.add_argument("--format", required=True, choices=["yolo", "coco", "imagefolder"], help="Input format")
    parser.add_argument("--images", help="Image directory (required for --format coco)")
    parser.add_argument(
        "--task",
        default=None,
        choices=["detection", "classification", "segmentation", "keypoints"],
        help="Override project task",
    )
    parser.add_argument("--copy", default="ingest", choices=["copy", "move", "hardlink", "reference", "ingest"], help="Asset copy mode")
    parser.add_argument("--class-map", help="Reserved for explicit class mapping files")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    if args.task and project.manifest.task != args.task:
        project.manifest.task = args.task
        project.save_manifest()

    if args.format == "coco":
        if not args.images:
            raise VisionPackError("--images is required when importing COCO (the directory holding the image files)")
        summary = CocoImporter(project, Path(args.source), Path(args.images), copy_mode=args.copy).run()
        label = "COCO"
    elif args.format == "imagefolder":
        summary = ImageFolderImporter(project, Path(args.source), copy_mode=args.copy).run()
        label = "ImageFolder"
    else:
        summary = YoloImporter(project, Path(args.source), copy_mode=args.copy).run()
        label = "YOLO"

    print(
        f"Imported {label} dataset: "
        f"{summary.assets} assets, {summary.annotations} annotations, {summary.objects} objects"
    )
    if summary.orphan_labels:
        print(f"Warnings: {summary.orphan_labels} label files had no matching image")
    if summary.classes_added:
        print("Classes were discovered and written to visionpack.yaml")
    return 0
