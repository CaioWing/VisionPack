from __future__ import annotations

import argparse

from visionpack.audit import AuditThresholds, audit_project
from visionpack.cli.output import emit_json
from visionpack.core.project import Project


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("audit", help="Label-health audit: suspicious (but valid) labels and class-balance risks")
    parser.add_argument("--min-box-px", type=float, default=None, help="Boxes thinner/shorter than this are degenerate (default: 8)")
    parser.add_argument("--duplicate-iou", type=float, default=None, help="Same-class IoU at/above this is a duplicate (default: 0.9)")
    parser.add_argument(
        "--max-aspect-ratio", type=float, default=None, help="Long/short side ratio beyond this is an outlier (default: 20)"
    )
    parser.add_argument("--imbalance-ratio", type=float, default=None, help="Most/least frequent class ratio that warns (default: 20)")
    parser.add_argument("--min-class-count", type=int, default=None, help="Classes with fewer objects are flagged rare (default: 10)")
    parser.add_argument("--limit", type=int, default=50, help="Max findings to print in human output (default: 50)")
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit non-zero when the audit finds anything (findings are advisory by default)",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    thresholds = AuditThresholds.from_project(
        project,
        min_box_px=args.min_box_px,
        duplicate_iou=args.duplicate_iou,
        max_aspect_ratio=args.max_aspect_ratio,
        imbalance_ratio=args.imbalance_ratio,
        min_class_count=args.min_class_count,
    )
    report = audit_project(project, thresholds)
    exit_code = 1 if (args.fail_on_findings and not report.ok) else 0

    if args.json:
        emit_json("audit", report.to_dict())
        return exit_code

    print(f"Audit: {len(report.findings)} finding(s) over {report.images_audited} image(s), {report.objects_audited} object(s)")
    for code, count in report.counts_by_code().items():
        print(f"  {code}: {count}")
    for finding in report.findings[: args.limit]:
        print(f"[{finding.code}] {finding.message}")
    if len(report.findings) > args.limit:
        print(f"... {len(report.findings) - args.limit} more findings (raise --limit or use --json)")
    if report.ok:
        print("No label-health findings.")
    return exit_code
