from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from visionpack.core.errors import VisionPackError

T = TypeVar("T")

# Tuned like rclone's low-level retries: enough attempts to ride out a rate
# limit or a dropped connection, short enough that a hard outage fails a sync
# in seconds, not minutes. Module-level so tests (and power users) can adjust.
MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 0.5

# Errors that retrying can never fix. FileNotFoundError doubles as fsspec's
# "no such key"; VisionPackError is our own diagnosis, already actionable.
_PERMANENT = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
    KeyboardInterrupt,
    VisionPackError,
)


def with_retries(operation: str, fn: Callable[[], T]) -> T:
    """Run ``fn``, retrying transient failures with exponential backoff.

    Object-store calls fail transiently all the time (throttling, dropped
    connections, 5xx) and every provider library spells those errors
    differently, so the classification is by exclusion: anything not provably
    permanent is worth retrying — every operation behind this wrapper is
    idempotent (reads, listings, full-object writes, server-side copies), so a
    retry can duplicate work but never corrupt state.

    When attempts run out the last error is wrapped in a
    :class:`VisionPackError` naming the operation, so the ingest loop records a
    clean per-object failure instead of crashing the whole sync on an exception
    type it doesn't know.
    """
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn()
        except _PERMANENT:
            raise
        except Exception as exc:  # provider-specific transient errors
            last = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(BASE_DELAY_SECONDS * (2**attempt))
    raise VisionPackError(f"{operation} failed after {MAX_ATTEMPTS} attempts: {last}") from last
