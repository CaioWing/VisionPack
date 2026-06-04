from __future__ import annotations

import argparse
import json

from visionpack.core.lock import project_lock
from visionpack.core.project import Project
from visionpack.snapshot import create_snapshot, list_snapshots, load_snapshot


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("snapshot", help="Manage dataset snapshots")
    nested = parser.add_subparsers(dest="snapshot_command", required=True)

    create = nested.add_parser("create", help="Create a snapshot")
    create.add_argument("-m", "--message", required=True, help="Snapshot message")
    create.set_defaults(func=run_create)

    list_parser = nested.add_parser("list", help="List snapshots")
    list_parser.set_defaults(func=run_list)

    show = nested.add_parser("show", help="Show snapshot details")
    show.add_argument("version", help="Snapshot version, e.g. v1")
    show.set_defaults(func=run_show)


def run_create(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        snapshot = create_snapshot(project, args.message)
    print(f"Created snapshot {snapshot['version']}: {snapshot['message']}")
    return 0


def run_list(args: argparse.Namespace) -> int:
    project = Project.open(".")
    snapshots = list_snapshots(project)
    for item in snapshots:
        stats = item.get("stats", {})
        counts = f"{stats.get('assets', '?')} imgs, {stats.get('objects', '?')} objs"
        print(f"{item['version']:<5} {item['created_at']}  {counts:<20}  {item['message']}")
    if not snapshots:
        print("No snapshots. Create one with: vp snapshot create -m <message>")
    return 0


def run_show(args: argparse.Namespace) -> int:
    project = Project.open(".")
    print(json.dumps(load_snapshot(project, args.version), indent=2, sort_keys=True))
    return 0
