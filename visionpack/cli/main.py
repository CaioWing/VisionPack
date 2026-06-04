from __future__ import annotations

import argparse
import sys

from visionpack.cli.commands import annotate, diff, export, fsck, import_, init, pack, snapshot, split, stats, sync, validate
from visionpack.core.errors import VisionPackError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vp", description="VisionPack DatasetOps CLI")
    parser.add_argument("--version", action="version", version="visionpack 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init.register(subparsers)
    import_.register(subparsers)
    sync.register(subparsers)
    validate.register(subparsers)
    fsck.register(subparsers)
    stats.register(subparsers)
    split.register(subparsers)
    snapshot.register(subparsers)
    diff.register(subparsers)
    export.register(subparsers)
    pack.register(subparsers)
    annotate.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except VisionPackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
