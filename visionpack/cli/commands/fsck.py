from __future__ import annotations

import argparse

from visionpack.core.project import Project
from visionpack.fsck import run_fsck


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("fsck", help="Check index <-> object-store integrity")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Re-hash every stored object to detect silent corruption (reads all bytes)",
    )
    parser.add_argument(
        "--no-orphans",
        action="store_true",
        help="Skip the scan for unreferenced objects in the store",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    report = run_fsck(project, deep=args.deep, check_orphans=not args.no_orphans)
    mode = "deep" if args.deep else "quick"
    print(
        f"fsck ({mode}): checked {report.checked_assets} assets, {report.checked_objects} objects "
        f"-> {len(report.errors)} errors, {len(report.warnings)} warnings"
    )
    for issue in report.issues[:50]:
        print(f"[{issue.severity}] {issue.code}: {issue.message}")
    if len(report.issues) > 50:
        print(f"... {len(report.issues) - 50} more")
    if report.ok:
        print("OK: dataset is consistent." if not report.warnings else "OK (with warnings).")
    return 0 if report.ok else 1
