from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from visionpack.core.models import utc_now

# A job body receives a progress callback `(done, total)` and returns a
# JSON-serializable result.
JobBody = Callable[[Callable[[int, int], None]], Any]


class JobBusyError(RuntimeError):
    """Another job is already running (the project lock allows one writer)."""


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    detail: str = ""
    state: str = "queued"  # queued | running | done | error
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "detail": self.detail,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """Runs one background job at a time and remembers the recent history.

    Serializing jobs mirrors the project lock (one writer per dataset): a busy
    manager answers immediately with :class:`JobBusyError` instead of queueing
    work the lock would make wait anyway. State transitions happen under a
    mutex, so polling `GET /api/jobs/{id}` always sees a consistent snapshot.
    """

    def __init__(self, on_finish: Callable[[Job], None] | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._mutex = threading.Lock()
        self._active: str | None = None
        self._on_finish = on_finish

    def start(self, kind: str, body: JobBody, detail: str = "") -> Job:
        with self._mutex:
            if self._active is not None and self._jobs[self._active].state in ("queued", "running"):
                raise JobBusyError(f"Job {self._active} ({self._jobs[self._active].kind}) is still running.")
            job = Job(id=f"job_{uuid.uuid4().hex[:12]}", kind=kind, detail=detail)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._active = job.id

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        def run() -> None:
            job.state = "running"
            try:
                job.result = body(progress)
                job.state = "done"
            except Exception as exc:  # surfaced to the client, never lost in a thread
                job.error = f"{exc}"
                job.state = "error"
                traceback.print_exc()
            finally:
                job.finished_at = utc_now()
                if self._on_finish is not None:
                    self._on_finish(job)

        threading.Thread(target=run, name=f"vp-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        return [self._jobs[job_id] for job_id in reversed(self._order[-limit:])]
