from __future__ import annotations

import argparse
from dataclasses import asdict

from visionpack.cli.output import emit_json
from visionpack.core.lock import project_lock
from visionpack.core.project import Project
from visionpack.progress import cli_progress
from visionpack.sources import plan_sources, sync_sources


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "sync",
        help="Reconcile the dataset with the sources declared in visionpack.yaml",
    )
    parser.add_argument("--source", help="Sync only this source (by name); default syncs all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what each source would ingest (found/matched/unmatched) without writing",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Concurrent transfers per source (default: 16+ for remote sources, CPU-derived for local)",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")

    if args.dry_run:
        plans = plan_sources(project, args.source)
        if args.json:
            emit_json("sync", {"dry_run": True, "plans": [asdict(plan) for plan in plans]})
            return 0
        for plan in plans:
            print(f"[{plan.name}] {plan.format}")
            print(f"  images: {plan.images_uri}")
            if plan.labels_uri:
                print(f"  labels: {plan.labels_uri}")
            print(
                f"  found {plan.images_found} images, {plan.labels_found} labels"
                f" -> {plan.matched} matched"
            )
            if plan.images_without_label:
                print(f"  {plan.images_without_label} images without a label")
            if plan.labels_without_image:
                print(f"  {plan.labels_without_image} labels without an image")
            print(f"  classes: {', '.join(plan.class_names) if plan.class_names else '(none discovered)'}")
        return 0

    # JSON mode keeps stdout to a single document: no progress bar.
    progress_factory = None if args.json else cli_progress
    with project_lock(project.root):
        summaries = sync_sources(project, args.source, progress_factory=progress_factory, max_workers=args.jobs)
    if args.json:
        total_failures = sum(len(summary.failures) for summary in summaries)
        emit_json(
            "sync",
            {
                "dry_run": False,
                "summaries": [asdict(summary) for summary in summaries],
                "total_assets_added": sum(summary.assets_added for summary in summaries),
                "total_failures": total_failures,
            },
        )
        return 1 if total_failures else 0
    total_added = 0
    total_failures = 0
    for summary in summaries:
        total_added += summary.assets_added
        total_failures += len(summary.failures)
        print(
            f"[{summary.name}] +{summary.assets_added} new assets "
            f"({summary.assets_existing} already present), "
            f"{summary.annotations} annotations, {summary.objects} objects"
        )
        if summary.classes_added:
            print(f"  {summary.classes_added} new classes merged into visionpack.yaml")
        if summary.images_without_label:
            print(f"  warning: {summary.images_without_label} images had no matching label")
        if summary.labels_without_image:
            print(f"  warning: {summary.labels_without_image} labels had no matching image")
        if summary.failures:
            print(f"  skipped {len(summary.failures)} unreadable/corrupt image(s):")
            for failure in summary.failures[:10]:
                print(f"    - {failure.path}: {failure.error}")
            if len(summary.failures) > 10:
                print(f"    ... {len(summary.failures) - 10} more")
    print(f"Synced {len(summaries)} source(s): {total_added} new assets total.")
    return 1 if total_failures else 0
