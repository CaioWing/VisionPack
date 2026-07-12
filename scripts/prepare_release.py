#!/usr/bin/env python3
"""Prepare a VisionPack release: bump the version, roll the changelog, run the
checks, build, and create the release commit + tag. Cross-platform port of the
former prepare-release.ps1.

    uv run python scripts/prepare_release.py 0.1.0
    uv run python scripts/prepare_release.py 0.1.0 --no-commit --no-tag  # files + checks only
    uv run python scripts/prepare_release.py 0.1.0 --dry-run             # print the plan

Publishing stays manual: push the branch and tag, create a draft GitHub
Release, and publishing it triggers the PyPI publish workflow.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9.\-]+)?$")


class ReleaseError(RuntimeError):
    pass


def run_step(label: str, command: list[str], *, dry_run: bool) -> None:
    print(f"==> {label}")
    print(f"    {' '.join(command)}")
    if dry_run:
        return
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise ReleaseError(f"Command failed: {' '.join(command)}")


def assert_clean_tracked_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ReleaseError("Unable to inspect git status.")
    if result.stdout.strip():
        raise ReleaseError("Tracked files have uncommitted changes. Commit or stash them before preparing a release.")


def assert_tag_does_not_exist(tag: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode == 0:
        raise ReleaseError(f"Tag {tag} already exists.")


def update_pyproject_version(version: str, *, dry_run: bool) -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"$', content)
    if match is None:
        raise ReleaseError("Could not find project version in pyproject.toml.")
    current = match.group(1)
    if current == version:
        raise ReleaseError(f"pyproject.toml is already at version {version}.")
    updated = content[: match.start()] + f'version = "{version}"' + content[match.end() :]
    print("==> Update pyproject.toml version")
    print(f"    {current} -> {version}")
    if not dry_run:
        PYPROJECT.write_text(updated, encoding="utf-8")


def update_changelog(version: str, *, dry_run: bool) -> None:
    if not CHANGELOG.exists():
        raise ReleaseError("CHANGELOG.md does not exist.")
    content = CHANGELOG.read_text(encoding="utf-8")
    if re.search(rf"(?m)^## \[{re.escape(version)}\]", content):
        raise ReleaseError(f"CHANGELOG.md already contains a section for {version}.")

    match = re.search(r"(?s)## \[Unreleased\]\r?\n(?P<body>.*?)(?=\r?\n## \[|\Z)", content)
    if match is None:
        raise ReleaseError("Could not find an Unreleased section in CHANGELOG.md.")

    body = match.group("body").strip()
    if not body or body == "- Nothing yet.":
        body = "- No changes documented."

    today = date.today().isoformat()
    replacement = f"## [Unreleased]\n\n- Nothing yet.\n\n## [{version}] - {today}\n\n{body}\n"
    updated = content[: match.start()] + replacement + content[match.end() :]
    print("==> Update CHANGELOG.md")
    print(f"    Move Unreleased notes to {version}")
    if not dry_run:
        CHANGELOG.write_text(updated, encoding="utf-8")


def check_build_artifacts(version: str) -> None:
    artifacts = sorted((REPO_ROOT / "dist").glob(f"visionpack-{version}*"))
    if not artifacts:
        raise ReleaseError(f"No dist artifacts found for version {version}.")
    run_step(
        "Validate distribution metadata",
        ["uvx", "twine", "check", *[str(path) for path in artifacts]],
        dry_run=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a VisionPack release (files, checks, build, commit, tag).")
    parser.add_argument("version", help="Release version, e.g. 0.1.0")
    parser.add_argument("--skip-checks", action="store_true", help="Skip ruff/mypy/tests")
    parser.add_argument("--no-commit", action="store_true", help="Do not create the release commit (implies --no-tag)")
    parser.add_argument("--no-tag", action="store_true", help="Do not create the release tag")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without changing anything")
    args = parser.parse_args()

    if not VERSION_PATTERN.match(args.version):
        parser.error(f"Invalid version {args.version!r}; expected MAJOR.MINOR.PATCH with an optional suffix.")
    if args.no_commit and not args.no_tag:
        parser.error("Use --no-tag with --no-commit; otherwise the tag would not include the release changes.")

    tag = f"v{args.version}"
    try:
        if not args.dry_run:
            assert_clean_tracked_worktree()
        assert_tag_does_not_exist(tag)
        update_pyproject_version(args.version, dry_run=args.dry_run)
        update_changelog(args.version, dry_run=args.dry_run)

        if not args.skip_checks:
            run_step("Run Ruff", ["uv", "run", "ruff", "check", "."], dry_run=args.dry_run)
            run_step("Run mypy", ["uv", "run", "mypy"], dry_run=args.dry_run)
            run_step(
                "Run unit tests",
                ["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests", "-q"],
                dry_run=args.dry_run,
            )

        run_step("Build source distribution and wheel", ["uv", "build"], dry_run=args.dry_run)
        if not args.dry_run:
            check_build_artifacts(args.version)

        if not args.no_commit:
            run_step("Stage release files", ["git", "add", "pyproject.toml", "CHANGELOG.md"], dry_run=args.dry_run)
            run_step("Create release commit", ["git", "commit", "-m", f"Release {tag}"], dry_run=args.dry_run)
        if not args.no_tag:
            run_step("Create release tag", ["git", "tag", tag], dry_run=args.dry_run)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Release {tag} is prepared.")
    print("Next steps:")
    print("  git push origin HEAD")
    print(f"  git push origin {tag}")
    print(f'  gh release create {tag} --draft --title "{tag}" --notes-file CHANGELOG.md')
    print("  Publish the GitHub Release when ready; the publish workflow will upload to PyPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
