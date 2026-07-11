from __future__ import annotations

import argparse
import json

from visionpack.cli.output import emit_json
from visionpack.core.lock import project_lock
from visionpack.core.project import Project
from visionpack.snapshot import create_snapshot, list_snapshots, load_snapshot, tag_snapshot, untag_snapshot


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("snapshot", help="Manage dataset snapshots")
    nested = parser.add_subparsers(dest="snapshot_command", required=True)

    create = nested.add_parser("create", help="Create a snapshot")
    create.add_argument("-m", "--message", required=True, help="Snapshot message")
    create.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    create.set_defaults(func=run_create)

    list_parser = nested.add_parser("list", help="List snapshots")
    list_parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    list_parser.set_defaults(func=run_list)

    show = nested.add_parser("show", help="Show snapshot details")
    show.add_argument("version", help="Snapshot version, e.g. v1")
    show.add_argument("--json", action="store_true", help="Print the machine-readable JSON envelope")
    show.set_defaults(func=run_show)

    tag = nested.add_parser("tag", help="Tag a snapshot for lineage (e.g. trained:run-812)")
    tag.add_argument("version", help="Snapshot version, e.g. v4")
    tag.add_argument("tag", help="Free-form tag; convention is key:value (trained:<run-id>)")
    tag.add_argument("--remove", action="store_true", help="Remove the tag instead of adding it")
    tag.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    tag.set_defaults(func=run_tag)


def run_create(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        snapshot = create_snapshot(project, args.message)
    if args.json:
        emit_json("snapshot.create", snapshot)
        return 0
    print(f"Created snapshot {snapshot['version']}: {snapshot['message']}")
    return 0


def run_list(args: argparse.Namespace) -> int:
    project = Project.open(".")
    snapshots = list_snapshots(project)
    if args.json:
        emit_json("snapshot.list", {"snapshots": snapshots})
        return 0
    for item in snapshots:
        stats = item.get("stats", {})
        counts = f"{stats.get('assets', '?')} imgs, {stats.get('objects', '?')} objs"
        tags = f"  [{', '.join(item['tags'])}]" if item.get("tags") else ""
        print(f"{item['version']:<5} {item['created_at']}  {counts:<20}  {item['message']}{tags}")
    if not snapshots:
        print("No snapshots. Create one with: vp snapshot create -m <message>")
    return 0


def run_tag(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        if args.remove:
            snapshot = untag_snapshot(project, args.version, args.tag)
        else:
            snapshot = tag_snapshot(project, args.version, args.tag)
    if args.json:
        emit_json("snapshot.tag", {"version": args.version, "tag": args.tag, "removed": args.remove, "tags": snapshot.get("tags", [])})
        return 0
    action = "Removed tag" if args.remove else "Tagged"
    print(f"{action} {args.version} {args.tag!r}; tags now: {', '.join(snapshot.get('tags', [])) or '(none)'}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    project = Project.open(".")
    snapshot = load_snapshot(project, args.version)
    if args.json:
        emit_json("snapshot.show", snapshot)
        return 0
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0
