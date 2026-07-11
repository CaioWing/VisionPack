from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from visionpack.cli.commands import (
    annotate,
    autolabel,
    diff,
    export,
    fsck,
    import_,
    init,
    pack,
    queue,
    serve,
    snapshot,
    split,
    stats,
    sync,
    validate,
)
from visionpack.cli.commands import (
    eval as eval_,
)
from visionpack.core.errors import VisionPackError


def _version() -> str:
    # Single source of truth is pyproject.toml; a source tree that was never
    # installed (no dist metadata) reports a dev placeholder instead of lying.
    try:
        return _package_version("visionpack")
    except PackageNotFoundError:
        return "0.0.0+dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vp", description="VisionPack DatasetOps CLI")
    parser.add_argument("--version", action="version", version=f"visionpack {_version()}")
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
    eval_.register(subparsers)
    autolabel.register(subparsers)
    queue.register(subparsers)
    serve.register(subparsers)
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
