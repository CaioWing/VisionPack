from __future__ import annotations

import argparse
from dataclasses import asdict

from visionpack.cli.output import emit_json
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
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    report = run_fsck(project, deep=args.deep, check_orphans=not args.no_orphans)
    mode = "deep" if args.deep else "quick"
    if args.json:
        emit_json(
            "fsck",
            {
                "ok": report.ok,
                "mode": mode,
                "checked_assets": report.checked_assets,
                "checked_objects": report.checked_objects,
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "issues": [asdict(issue) for issue in report.issues],
            },
        )
        return 0 if report.ok else 1
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
