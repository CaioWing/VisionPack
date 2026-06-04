from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

# (completed, total) -> None. Total may be 0/unknown until known.
ProgressCallback = Callable[[int, int], None]


@contextmanager
def cli_progress(description: str) -> Iterator[ProgressCallback | None]:
    """Yield a progress callback backed by a rich bar — but only on a real
    terminal. In CI / piped logs it yields ``None`` (a no-op for the library), so
    long runs don't spam the log with animation control codes.
    """
    if not sys.stderr.isatty():
        yield None
        return
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=_console(),
    ) as progress:
        task = progress.add_task(description, total=None)

        def update(completed: int, total: int) -> None:
            progress.update(task, completed=completed, total=total or None)

        yield update


def _console():
    from rich.console import Console

    # Render to stderr so progress never pollutes stdout (which scripts may parse).
    return Console(stderr=True)
