from __future__ import annotations

import argparse

from visionpack.cli.output import emit_json
from visionpack.core.project import Project
from visionpack.diff import diff_snapshots
from visionpack.drift import drift_from_stats


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("diff", help="Diff two snapshots")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Also report class-distribution drift (per-class share deltas, KL/JS divergence)",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    result = diff_snapshots(project, args.left, args.right)
    drift = None
    if args.drift:
        # Reuse the stats the diff already loaded; no second snapshot read.
        drift = drift_from_stats(result["stats_before"], result["stats_after"], left=args.left, right=args.right)
    if args.json:
        payload = {"left": args.left, "right": args.right, **result}
        if drift is not None:
            payload["drift"] = drift
        emit_json("diff", payload)
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
    if drift is not None:
        _print_drift(drift)
    return 0


def _print_drift(drift: dict) -> None:
    print(
        f"\nDrift {drift['from']} -> {drift['to']}: "
        f"{drift['images_before']} -> {drift['images_after']} images, "
        f"{drift['objects_before']} -> {drift['objects_after']} objects"
    )
    if drift["kl_divergence"] is not None:
        print(f"KL divergence (after || before): {drift['kl_divergence']}")
        print(f"JS divergence (symmetric, max ln2 = 0.693): {drift['js_divergence']}")
    if not drift["classes"]:
        print("No labeled objects in either snapshot; nothing to compare.")
        return
    print(f"{'class':<24} {'before':>8} {'after':>8} {'delta':>7} {'share delta':>12}")
    for item in drift["classes"][:20]:
        print(
            f"{item['class_id']:<24} {item['before']:>8} {item['after']:>8} "
            f"{item['delta']:>+7} {item['share_delta']:>+11.2%}"
        )
    if len(drift["classes"]) > 20:
        print(f"... {len(drift['classes']) - 20} more classes (use --json for all)")
