from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("pack", help="Pack a dataset for archive, training, or review")
    parser.add_argument("--profile", required=True, help="Pack profile name")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    print("vp pack is scaffolded but not implemented in this MVP slice yet.")
    print("Implemented commands: init, import, validate, stats, snapshot, diff, export.")
    return 1
