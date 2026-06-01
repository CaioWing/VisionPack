from __future__ import annotations

import argparse
import json

from visionpack.core.project import Project
from visionpack.diff import diff_snapshots


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("diff", help="Diff two snapshots")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    result = diff_snapshots(project, args.left, args.right)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"Diff {args.left} -> {args.right}")
    for key in (
        "assets_added",
        "assets_removed",
        "annotations_added",
        "annotations_removed",
        "annotations_modified",
        "classes_added",
        "classes_removed",
    ):
        print(f"{key}: {len(result[key])}")
    print(f"splits_changed: {result['splits_changed']}")
    return 0
