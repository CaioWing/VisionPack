from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from visionpack.core.project import Project
from visionpack.validation import validate_project


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validate", help="Validate indexed assets and annotations")
    parser.add_argument("--strict", action="store_true", help="Treat missing annotations as errors")
    parser.add_argument("--fix", action="store_true", help="Reserved for future automatic fixes")
    parser.add_argument("--report", help="Write JSON validation report")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    report = validate_project(project, strict=args.strict)
    print(f"Validation: {len(report.errors)} errors, {len(report.warnings)} warnings")
    for issue in report.issues[:50]:
        print(f"[{issue.severity}] {issue.code}: {issue.message}")
    if len(report.issues) > 50:
        print(f"... {len(report.issues) - 50} more issues")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(issue) for issue in report.issues], indent=2), encoding="utf-8")
        print(f"Report written: {path}")
    return 0 if report.ok else 1
