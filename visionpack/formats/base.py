from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class IngestFailure:
    """An image that could not be ingested (corrupt, unreadable, missing).

    Collected instead of aborting the whole batch, so one bad file in 100k
    doesn't sink the run.
    """

    path: str
    error: str


@dataclass(slots=True)
class ImportSummary:
    assets: int = 0
    annotations: int = 0
    objects: int = 0
    orphan_labels: int = 0
    classes_added: int = 0
    failures: list[IngestFailure] = field(default_factory=list)
