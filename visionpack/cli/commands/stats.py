from __future__ import annotations

import argparse
import json

from visionpack.core.project import Project
from visionpack.stats import collect_stats


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("stats", help="Show dataset statistics")
    parser.add_argument("--by", choices=["class", "split"], help="Focus output on one dimension")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--html", help="Reserved for HTML reports")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    stats = collect_stats(project)
    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    if args.by == "class":
        for class_id, count in stats["class_distribution"].items():
            print(f"{class_id}: {count}")
        return 0
    print(f"Images: {stats['assets']}")
    print(f"Annotations: {stats['annotations']}")
    print(f"Objects: {stats['objects']}")
    print(f"Classes: {stats['classes']}")
    print(f"Images without annotations: {stats['images_without_annotations']}")
    print(f"Average labels per image: {stats['avg_labels_per_image']}")
    return 0
