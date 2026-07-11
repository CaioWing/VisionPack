"""Distribution drift between two snapshots (``vp diff --drift``).

A dataset that grows is supposed to change; what needs watching is *how* its
class distribution changes. A new batch of images that doubles one class and
starves another silently shifts what the next model optimizes for — metrics
move and nobody knows whether the model or the data changed.

This module compares the class distributions captured in two snapshots'
``stats`` blocks (no inventory rehydration needed — snapshot stats are computed
at create time) and reports:

- **per-class deltas**: object counts and distribution share before/after, so
  the classes driving the shift are named, not just scored;
- **divergence scores**: Kullback–Leibler (after vs. before, additively
  smoothed so new/vanished classes stay finite) and Jensen–Shannon divergence
  (symmetric, bounded to [0, ln 2]) as single drift numbers a CI job can
  threshold.

Everything derives from data already frozen in the snapshots, so drift between
``v1`` and ``v2`` is reproducible forever.
"""

from __future__ import annotations

import math
from typing import Any

from visionpack.core.project import Project
from visionpack.snapshot import load_snapshot

# Additive smoothing applied to both distributions before KL: keeps a class
# that appears (or disappears) between snapshots from producing infinity while
# barely perturbing well-populated classes.
_SMOOTHING = 0.5


def drift_between(project: Project, left: str, right: str) -> dict[str, Any]:
    """Class-distribution drift from snapshot ``left`` to snapshot ``right``."""
    old = load_snapshot(project, left)
    new = load_snapshot(project, right)
    return drift_from_stats(old.get("stats", {}), new.get("stats", {}), left=left, right=right)


def drift_from_stats(
    old_stats: dict[str, Any], new_stats: dict[str, Any], *, left: str = "before", right: str = "after"
) -> dict[str, Any]:
    old_counts = {str(k): int(v) for k, v in old_stats.get("class_distribution", {}).items()}
    new_counts = {str(k): int(v) for k, v in new_stats.get("class_distribution", {}).items()}
    classes = sorted(set(old_counts) | set(new_counts))
    old_total = sum(old_counts.values())
    new_total = sum(new_counts.values())

    per_class: list[dict[str, Any]] = []
    for class_id in classes:
        before = old_counts.get(class_id, 0)
        after = new_counts.get(class_id, 0)
        share_before = before / old_total if old_total else 0.0
        share_after = after / new_total if new_total else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "before": before,
                "after": after,
                "delta": after - before,
                "share_before": round(share_before, 6),
                "share_after": round(share_after, 6),
                "share_delta": round(share_after - share_before, 6),
            }
        )
    # Biggest distribution movers first, so the head of the list is the story.
    per_class.sort(key=lambda item: (-abs(item["share_delta"]), item["class_id"]))

    return {
        "from": left,
        "to": right,
        "classes": per_class,
        "objects_before": old_total,
        "objects_after": new_total,
        "images_before": int(old_stats.get("assets", 0)),
        "images_after": int(new_stats.get("assets", 0)),
        "kl_divergence": _kl(old_counts, new_counts, classes),
        "js_divergence": _js(old_counts, new_counts, classes),
    }


def _distribution(counts: dict[str, int], classes: list[str]) -> list[float]:
    smoothed = [counts.get(class_id, 0) + _SMOOTHING for class_id in classes]
    total = sum(smoothed)
    return [value / total for value in smoothed]


def _kl(old_counts: dict[str, int], new_counts: dict[str, int], classes: list[str]) -> float | None:
    """KL(after || before), smoothed. ``None`` when either side has no labels."""
    if not classes or not sum(old_counts.values()) or not sum(new_counts.values()):
        return None
    p = _distribution(new_counts, classes)
    q = _distribution(old_counts, classes)
    return round(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True)), 6)


def _js(old_counts: dict[str, int], new_counts: dict[str, int], classes: list[str]) -> float | None:
    """Jensen–Shannon divergence: symmetric, bounded to [0, ln 2]."""
    if not classes or not sum(old_counts.values()) or not sum(new_counts.values()):
        return None
    p = _distribution(new_counts, classes)
    q = _distribution(old_counts, classes)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q, strict=True)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b, strict=True))

    return round((kl(p, m) + kl(q, m)) / 2, 6)
