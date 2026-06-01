from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.core.project import Project
from visionpack.packing import pack_archive


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("pack", help="Pack a dataset for archive, training, or review")
    parser.add_argument("--profile", required=True, help="Pack profile name")
    parser.add_argument("--output", help="Output archive path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    if args.profile != "archive":
        print("Only archive packing is implemented in this MVP slice.")
        print("Use: vp pack --profile archive")
        return 1
    summary = pack_archive(project, output=Path(args.output) if args.output else None, profile_name=args.profile)
    print(
        f"Packed archive: {summary.path} "
        f"({summary.format}, {summary.files} files, {summary.assets} assets, {summary.size_bytes} bytes)"
    )
    return 0
