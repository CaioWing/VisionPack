from __future__ import annotations

import argparse

from visionpack.cli.output import emit_json
from visionpack.core.lock import project_lock
from visionpack.core.project import Project
from visionpack.split import create_split, lock_split


def _split_payload(split) -> dict:  # noqa: ANN001
    return {
        "id": split.id,
        "strategy": split.strategy,
        "locked": split.locked,
        "sets": {name: len(ids) for name, ids in split.sets.items()},
    }


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("split", help="Create and manage deterministic, versioned splits")
    nested = parser.add_subparsers(dest="split_command", required=True)

    create = nested.add_parser("create", help="Create a deterministic train/val/test split")
    create.add_argument("--train", type=float, default=0.8, help="Train fraction (default 0.8)")
    create.add_argument("--val", type=float, default=0.1, help="Validation fraction (default 0.1)")
    create.add_argument("--test", type=float, default=0.1, help="Test fraction (default 0.1)")
    create.add_argument(
        "--strategy",
        choices=["stratified", "random", "hash"],
        default="stratified",
        help="stratified=balanced per class (default), random=exact global ratios, hash=stable as data grows",
    )
    create.add_argument("--by", choices=["class"], default="class", help="Stratification key (stratified strategy)")
    create.add_argument("--seed", type=int, default=0, help="Seed mixed into the content hash for reproducibility")
    create.add_argument("--id", dest="split_id", default="default", help="Split id (default: 'default')")
    create.add_argument("--force", action="store_true", help="Overwrite even if the split is locked")
    create.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    create.set_defaults(func=run_create)

    lock = nested.add_parser("lock", help="Lock a split so it cannot be changed")
    lock.add_argument("--id", dest="split_id", default="default", help="Split id (default: 'default')")
    lock.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    lock.set_defaults(func=run_lock)

    list_parser = nested.add_parser("list", help="List splits")
    list_parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    list_parser.set_defaults(func=run_list)

    show = nested.add_parser("show", help="Show set sizes for a split")
    show.add_argument("--id", dest="split_id", default="default", help="Split id (default: 'default')")
    show.add_argument("--json", action="store_true", help="Print a machine-readable JSON result")
    show.set_defaults(func=run_show)


def run_create(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        split = create_split(
            project,
            train=args.train,
            val=args.val,
            test=args.test,
            strategy=args.strategy,
            seed=args.seed,
            split_id=args.split_id,
            by=args.by,
            force=args.force,
        )
    if args.json:
        emit_json("split.create", {**_split_payload(split), "seed": args.seed})
        return 0
    sizes = ", ".join(f"{name}={len(ids)}" for name, ids in split.sets.items())
    print(f"Created split {split.id!r} (strategy={split.strategy}, seed={args.seed}): {sizes}")
    return 0


def run_lock(args: argparse.Namespace) -> int:
    project = Project.open(".")
    with project_lock(project.root):
        split = lock_split(project, args.split_id)
    if args.json:
        emit_json("split.lock", _split_payload(split))
        return 0
    print(f"Locked split {split.id!r}. It will be captured as-is in snapshots.")
    return 0


def run_list(args: argparse.Namespace) -> int:
    project = Project.open(".")
    splits = project.index.splits()
    if args.json:
        emit_json("split.list", {"splits": [_split_payload(split) for split in splits]})
        return 0
    for split in splits:
        sizes = ", ".join(f"{name}={len(ids)}" for name, ids in split.sets.items())
        lock = " [locked]" if split.locked else ""
        print(f"{split.id}  strategy={split.strategy}{lock}  {sizes}")
    if not splits:
        print("No splits. Create one with: vp split create")
    return 0


def run_show(args: argparse.Namespace) -> int:
    project = Project.open(".")
    split = next((item for item in project.index.splits() if item.id == args.split_id), None)
    if split is None:
        if args.json:
            emit_json("split.show", {"id": args.split_id, "found": False})
            return 1
        print(f"No split named {args.split_id!r}. Create one with: vp split create")
        return 1
    if args.json:
        emit_json("split.show", {**_split_payload(split), "found": True, "asset_ids": split.sets})
        return 0
    print(f"Split {split.id!r}  strategy={split.strategy}  locked={split.locked}")
    for name, ids in split.sets.items():
        print(f"  {name}: {len(ids)} images")
    return 0
