from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ImportSummary:
    assets: int = 0
    annotations: int = 0
    objects: int = 0
    orphan_labels: int = 0
    classes_added: int = 0
