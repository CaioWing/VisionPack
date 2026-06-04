from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from visionpack.core.errors import VisionPackError


@contextmanager
def project_lock(root: Path) -> Iterator[None]:
    """Hold an exclusive lock on a dataset for the duration of a mutating command.

    Two ``vp`` processes writing the same index concurrently could lose updates
    (each loads, mutates, and writes back). This takes an OS advisory lock on
    ``.vp/lock`` — non-blocking, so a second writer fails fast with a clear
    message instead of corrupting state. The lock is tied to the open file
    handle, so the OS releases it automatically if the process dies; there are no
    stale lock files to clean up by hand.
    """
    lock_path = root / ".vp" / "lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(fd, lock_path)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("ascii"))
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


def _acquire(fd: int, lock_path: Path) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise VisionPackError(
            f"Another vp process is modifying this dataset ({lock_path} is locked). "
            "Wait for it to finish and try again."
        ) from exc


def _release(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
