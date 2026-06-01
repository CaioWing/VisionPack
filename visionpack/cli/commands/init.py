from __future__ import annotations

import argparse

from visionpack.core.project import Project


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("init", help="Initialize a VisionPack project")
    parser.add_argument("path", nargs="?", default=".", help="Project directory")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--task", default="detection", choices=["detection"], help="Computer vision task")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.init(args.path, name=args.name, task=args.task)
    print(f"Initialized VisionPack dataset at {project.root}")
    print(f"Manifest: {project.manifest_path}")
    return 0
