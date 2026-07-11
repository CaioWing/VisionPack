from __future__ import annotations

import argparse
from pathlib import Path

from visionpack.cli.output import emit_json
from visionpack.core.project import Project
from visionpack.curation import rank_for_annotation
from visionpack.predictions import FORMATS, load_predictions


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("queue", help="Rank images by how much a human label would help (active learning)")
    parser.add_argument("--predictions", help="Optional predictions file/dir to rank by model uncertainty and disagreement")
    parser.add_argument("--format", choices=list(FORMATS), default="auto", help="Predictions format (default: auto-detect)")
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Also audit labeled images for ground-truth/prediction disagreement (requires --predictions)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Show at most this many images (default: 50)")
    parser.add_argument("--json", action="store_true", help="Print the full ranking as JSON")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    predictions = None
    if args.predictions:
        predictions = load_predictions(project, Path(args.predictions), fmt=args.format)
    elif args.include_labeled:
        print("error: --include-labeled needs --predictions (disagreement is measured against a model).")
        return 2

    ranked = rank_for_annotation(project, predictions, include_labeled=args.include_labeled)
    if args.json:
        shown = ranked[: args.limit] if args.limit else ranked
        emit_json("queue", {"total": len(ranked), "items": shown})
        return 0

    if not ranked:
        print("Queue is empty: every image is labeled and (if predictions were given) the model agrees with the labels.")
        return 0
    shown = ranked[: args.limit] if args.limit else ranked
    print(f"{len(ranked)} image(s) queued for annotation; showing {len(shown)}:")
    for item in shown:
        print(f"{item['score']:>7.4f}  {item['asset_id']}  {item['path']}")
        for reason in item["reasons"]:
            print(f"         - {reason}")
    return 0
