from __future__ import annotations

from dataclasses import dataclass, field

# Characters that would let a name escape (or misbehave in) an export tree when
# used as a single path component: separators, traversal, and Windows-reserved.
_UNSAFE_COMPONENT_CHARS = str.maketrans({ch: "_" for ch in '/\\:*?"<>|\0'})


def safe_path_component(name: str, fallback: str = "unnamed") -> str:
    """Make ``name`` safe to use as one directory/file name inside an export.

    Class names come from *imported data* (folder names, COCO ``categories``,
    ``classes.txt``), so a name like ``../../x`` or ``a/b`` must never be able
    to place files outside — or in unexpected subtrees of — the export
    directory. Separators and reserved characters are replaced, traversal names
    collapse to the fallback, and the result is never empty.
    """
    cleaned = name.translate(_UNSAFE_COMPONENT_CHARS).strip()
    if cleaned.strip(".") == "":  # "", ".", "..", "..." — traversal or hidden-empty
        return fallback
    return cleaned


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
