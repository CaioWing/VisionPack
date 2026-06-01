from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("annotate", help="Prepare or ingest annotation batches")
    nested = parser.add_subparsers(dest="annotate_command", required=True)

    prepare = nested.add_parser("prepare", help="Prepare images for annotation")
    prepare.add_argument("--target", required=True)
    prepare.set_defaults(func=run_scaffold)

    ingest = nested.add_parser("ingest", help="Ingest reviewed annotations")
    ingest.add_argument("source")
    ingest.add_argument("--format", required=True)
    ingest.set_defaults(func=run_scaffold)

    review = nested.add_parser("review", help="Review annotations")
    review.set_defaults(func=run_scaffold)


def run_scaffold(args: argparse.Namespace) -> int:
    print("vp annotate is scaffolded but not implemented in this MVP slice yet.")
    return 1
