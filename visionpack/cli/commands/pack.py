from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from visionpack.cli.output import emit_json
from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project
from visionpack.packing import pack_archive, pack_training


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("pack", help="Pack a dataset for archive, training, or review")
    parser.add_argument("--profile", required=True, help="Pack profile name (from visionpack.yaml)")
    parser.add_argument("--output", help="Output path (archive file, or directory for WebDataset shards)")
    parser.add_argument(
        "--split",
        nargs="?",
        const="default",
        default=None,
        help="For training packs: emit per-set shards from this split (default 'default' when given without a value)",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    profile = project.manifest.pack_profiles.get(args.profile)
    if profile is None:
        raise VisionPackError(f"Pack profile not found in visionpack.yaml: {args.profile}")

    fmt = str(profile.get("format", "tar.zst"))
    output = Path(args.output) if args.output else None

    if fmt == "webdataset":
        summary = pack_training(project, output=output, profile_name=args.profile, split_id=args.split)
        if args.json:
            emit_json("pack", {"profile": args.profile, "format": fmt, **asdict(summary)})
            return 0
        sets = ", ".join(f"{name}={count}" for name, count in summary.sets.items())
        print(
            f"Packed WebDataset to {summary.path} "
            f"({summary.shards} shards, {summary.samples} samples; {sets})"
        )
        if summary.skipped:
            print(f"Skipped {summary.skipped} assets not assigned to any set in split {args.split!r}")
        return 0

    if fmt in {"tar", "tar.zst"}:
        archive = pack_archive(project, output=output, profile_name=args.profile)
        if args.json:
            emit_json("pack", {"profile": args.profile, **asdict(archive)})
            return 0
        print(
            f"Packed archive: {archive.path} "
            f"({archive.format}, {archive.files} files, {archive.assets} assets, {archive.size_bytes} bytes)"
        )
        return 0

    raise VisionPackError(
        f"Pack profile {args.profile!r} uses unsupported format {fmt!r}. "
        "Supported: 'webdataset' (training), 'tar'/'tar.zst' (archive)."
    )
