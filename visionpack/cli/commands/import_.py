from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from visionpack.cli.output import emit_json
from visionpack.core.errors import VisionPackError
from visionpack.core.lock import project_lock
from visionpack.core.project import Project
from visionpack.formats.classification import ImageFolderImporter
from visionpack.formats.coco import CocoImporter
from visionpack.formats.yolo import YoloImporter
from visionpack.progress import cli_progress


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
    parser.add_argument("--name", help="Name to record this source under in visionpack.yaml")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not add this import as a source in visionpack.yaml (use for one-off/throwaway imports)",
    )
    parser.add_argument("--class-map", help="Reserved for explicit class mapping files")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        return _run_locked(project, args)


def _run_locked(project: Project, args: argparse.Namespace) -> int:
    if args.task and project.manifest.task != args.task:
        project.manifest.task = args.task
        project.save_manifest()

    if args.format == "coco":
        if not args.images:
            raise VisionPackError("--images is required when importing COCO (the directory holding the image files)")
        importer = CocoImporter(project, Path(args.source), Path(args.images), copy_mode=args.copy)
        label = "COCO"
    elif args.format == "imagefolder":
        importer = ImageFolderImporter(project, Path(args.source), copy_mode=args.copy)
        label = "ImageFolder"
    else:
        importer = YoloImporter(project, Path(args.source), copy_mode=args.copy)
        label = "YOLO"

    if args.json:
        summary = importer.run()
        recorded = None if args.no_record else _record_source(project, args)
        emit_json(
            "import",
            {
                "format": args.format,
                "assets": summary.assets,
                "annotations": summary.annotations,
                "objects": summary.objects,
                "classes_added": summary.classes_added,
                "orphan_labels": summary.orphan_labels,
                "recorded_source": recorded,
                "failures": [asdict(failure) for failure in summary.failures],
            },
        )
        return 1 if summary.failures else 0

    with cli_progress(f"Importing {label}") as callback:
        summary = importer.run(progress=callback)

    print(
        f"Imported {label} dataset: "
        f"{summary.assets} assets, {summary.annotations} annotations, {summary.objects} objects"
    )
    if summary.orphan_labels:
        print(f"Warnings: {summary.orphan_labels} label files had no matching image")
    if summary.classes_added:
        print("Classes were discovered and written to visionpack.yaml")

    if not args.no_record:
        recorded = _record_source(project, args)
        if recorded:
            print(f"Recorded source {recorded!r} in visionpack.yaml (re-pull later with `vp sync --source {recorded}`)")

    if summary.failures:
        _report_failures(summary.failures)
        return 1
    return 0


def _report_failures(failures: list) -> None:
    print(f"Skipped {len(failures)} unreadable/corrupt image(s):")
    for failure in failures[:20]:
        print(f"  - {failure.path}: {failure.error}")
    if len(failures) > 20:
        print(f"  ... {len(failures) - 20} more")


def _record_source(project: Project, args: argparse.Namespace) -> str | None:
    """Append this import as a declared source so the manifest stays the source of
    truth and the data can be re-pulled with `vp sync`. Skips silently when an
    equivalent source is already declared (re-importing the same path)."""
    entry = _source_entry(project, args)
    # Dedupe on location identity only: a reloaded source carries pydantic default
    # keys (class_map/credentials={}) the fresh entry lacks, so compare the parts
    # that actually identify where the data comes from.
    identity = _identity(entry)
    if any(_identity(declared) == identity for declared in project.manifest.sources):
        return None
    # name first for readability in the manifest
    entry = {"name": _unique_name(project, args), **entry}
    project.manifest.sources.append(entry)
    project.save_manifest()
    return entry["name"]


def _identity(source: dict) -> tuple:
    return (source.get("format"), source.get("root"), source.get("images"), source.get("labels"))


def _source_entry(project: Project, args: argparse.Namespace) -> dict[str, str]:
    if args.format == "coco":
        return {
            "format": "coco",
            "images": _rel(project, args.images),
            "labels": _rel(project, args.source),
            "copy": args.copy,
        }
    fmt = "imagefolder" if args.format == "imagefolder" else "yolo"
    return {"format": fmt, "root": _rel(project, args.source), "copy": args.copy}


def _rel(project: Project, path: str) -> str:
    """A project-relative, posix-style location, so the manifest stays portable."""
    resolved = Path(path).resolve()
    try:
        rel = Path(os.path.relpath(resolved, project.root)).as_posix()
    except ValueError:  # different drive on Windows
        return resolved.as_posix()
    return rel if rel.startswith((".", "/")) else f"./{rel}"


def _unique_name(project: Project, args: argparse.Namespace) -> str:
    base = Path(args.name or Path(args.source).stem or args.format).name or args.format
    taken = {source.get("name") for source in project.manifest.sources}
    if base not in taken:
        return base
    suffix = 2
    while f"{base}-{suffix}" in taken:
        suffix += 1
    return f"{base}-{suffix}"
