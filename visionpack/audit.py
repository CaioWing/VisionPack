"""Label-health audit (``vp audit``): find labels that are *suspicious*, not invalid.

``vp validate`` catches labels that are wrong by construction (zero area, out of
bounds, unknown class). This audit targets the quieter failure mode: labels that
pass validation but usually mean an annotation mistake or a dataset-level risk —

- **duplicate boxes**: two same-class objects on one image with near-total
  overlap (double-click, double-import, or a stale autolabel pass);
- **degenerate boxes**: valid but tiny — usually annotation noise the model can
  never learn from;
- **edge-pinned boxes**: boxes pinned to two or more image borders, or covering
  nearly the whole image — typical of clipping bugs and auto-label artifacts;
- **aspect-ratio outliers**: extreme sliver boxes (a 40:1 "person") that are
  almost always drawing errors;
- **class imbalance**: a dataset-level warning when the most frequent class
  dwarfs the rarest, plus per-class warnings for classes too rare to learn.

Every finding is advisory (a warning, never an error): the audit is a review
queue, not a gate — though ``vp audit --fail-on-findings`` lets CI treat it as
one. Thresholds come from ``validation.audit`` in ``visionpack.yaml`` and can be
overridden per run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from visionpack.core.models import Annotation, Asset, BBox
from visionpack.core.project import Project
from visionpack.eval import bbox_iou

# Defaults, overridable via manifest `validation.audit` and the CLI flags.
DEFAULT_MIN_BOX_PX = 8.0  # boxes thinner/shorter than this are degenerate
DEFAULT_DUPLICATE_IOU = 0.9  # same-class overlap at/above this is a duplicate
DEFAULT_MAX_ASPECT_RATIO = 20.0  # long-side / short-side beyond this is an outlier
DEFAULT_EDGE_TOLERANCE_PX = 1.0  # how close to a border counts as touching it
DEFAULT_COVERS_IMAGE_RATIO = 0.98  # box area / image area at/above this covers it
DEFAULT_IMBALANCE_RATIO = 20.0  # most-frequent / least-frequent class warning
DEFAULT_MIN_CLASS_COUNT = 10  # classes with fewer instances are flagged rare


@dataclass(slots=True)
class AuditThresholds:
    """Tunable limits for every audit check. See the module docstring."""

    min_box_px: float = DEFAULT_MIN_BOX_PX
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO
    edge_tolerance_px: float = DEFAULT_EDGE_TOLERANCE_PX
    covers_image_ratio: float = DEFAULT_COVERS_IMAGE_RATIO
    imbalance_ratio: float = DEFAULT_IMBALANCE_RATIO
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT

    @classmethod
    def from_project(cls, project: Project, **overrides: Any) -> AuditThresholds:
        """Manifest ``validation.audit`` values, overlaid by explicit overrides.

        Precedence (lowest to highest): built-in defaults, ``visionpack.yaml``,
        keyword overrides (the CLI flags). ``None`` overrides are ignored so
        callers can pass optional flags straight through.
        """
        config = dict(project.manifest.validation.get("audit", {}))
        config.update({key: value for key, value in overrides.items() if value is not None})
        known = {f: config[f] for f in cls.__dataclass_fields__ if f in config}  # ignore unknown keys
        return cls(**known)


@dataclass(slots=True)
class AuditFinding:
    code: str
    message: str
    asset_id: str | None = None
    path: str | None = None
    class_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "asset_id": self.asset_id, "path": self.path, "class_id": self.class_id}


@dataclass(slots=True)
class AuditReport:
    findings: list[AuditFinding]
    images_audited: int = 0
    objects_audited: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def counts_by_code(self) -> dict[str, int]:
        counts = Counter(finding.code for finding in self.findings)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": len(self.findings),
            "by_code": self.counts_by_code(),
            "images_audited": self.images_audited,
            "objects_audited": self.objects_audited,
            "class_counts": dict(sorted(self.class_counts.items())),
            "items": [finding.to_dict() for finding in self.findings],
        }


def audit_project(project: Project, thresholds: AuditThresholds | None = None) -> AuditReport:
    """Run every label-health check over the whole dataset in one streamed pass."""
    limits = thresholds or AuditThresholds.from_project(project)
    findings: list[AuditFinding] = []
    class_counts: Counter[str] = Counter()
    images = 0
    objects = 0

    for asset, annotation in project.index.iter_assets_with_annotations():
        images += 1
        if annotation is None or not annotation.objects:
            continue
        objects += len(annotation.objects)
        class_counts.update(obj.class_id for obj in annotation.objects)
        findings.extend(_audit_image(asset, annotation, limits))

    findings.extend(_audit_class_balance(class_counts, limits))
    return AuditReport(findings=findings, images_audited=images, objects_audited=objects, class_counts=dict(class_counts))


# --- per-image checks ---------------------------------------------------------


def _audit_image(asset: Asset, annotation: Annotation, limits: AuditThresholds) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    boxed = [(obj, obj.bbox) for obj in annotation.objects if obj.bbox is not None]

    for obj, box in boxed:
        findings.extend(_audit_box(asset, obj.class_id, box, limits))

    # Duplicate detection is O(n²) per image over same-class pairs — n is the
    # object count of one image, so this stays trivial even on dense scenes.
    for i in range(len(boxed)):
        for j in range(i + 1, len(boxed)):
            obj_a, box_a = boxed[i]
            obj_b, box_b = boxed[j]
            if obj_a.class_id != obj_b.class_id:
                continue
            iou = bbox_iou(box_a, box_b)
            if iou >= limits.duplicate_iou:
                findings.append(
                    AuditFinding(
                        "box.duplicate",
                        f"Two {obj_a.class_id!r} boxes overlap at IoU {iou:.2f} in {asset.original_path} — "
                        "likely the same object labeled twice",
                        asset.id,
                        asset.original_path,
                        obj_a.class_id,
                    )
                )
    return findings


def _audit_box(asset: Asset, class_id: str, box: BBox, limits: AuditThresholds) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if box.width <= 0 or box.height <= 0:
        return findings  # invalid, vp validate's territory

    if box.width < limits.min_box_px or box.height < limits.min_box_px:
        findings.append(
            AuditFinding(
                "box.degenerate",
                f"Tiny {class_id!r} box ({box.width:.0f}x{box.height:.0f} px, threshold {limits.min_box_px:.0f}) in {asset.original_path}",
                asset.id,
                asset.original_path,
                class_id,
            )
        )

    ratio = max(box.width / box.height, box.height / box.width)
    if ratio > limits.max_aspect_ratio:
        findings.append(
            AuditFinding(
                "box.aspect_outlier",
                f"Extreme aspect ratio {ratio:.1f}:1 for {class_id!r} box in {asset.original_path}",
                asset.id,
                asset.original_path,
                class_id,
            )
        )

    if asset.width > 0 and asset.height > 0:
        coverage = (box.width * box.height) / (asset.width * asset.height)
        if coverage >= limits.covers_image_ratio:
            findings.append(
                AuditFinding(
                    "box.covers_image",
                    f"{class_id!r} box covers {coverage:.0%} of {asset.original_path} — whole-image boxes are usually labeling artifacts",
                    asset.id,
                    asset.original_path,
                    class_id,
                )
            )
        else:
            tol = limits.edge_tolerance_px
            edges = sum(
                (
                    box.x <= tol,
                    box.y <= tol,
                    box.x + box.width >= asset.width - tol,
                    box.y + box.height >= asset.height - tol,
                )
            )
            # One touched border is normal (an object leaving the frame); two or
            # more usually means clipping bugs or coordinate-space mistakes.
            if edges >= 2:
                findings.append(
                    AuditFinding(
                        "box.edge_pinned",
                        f"{class_id!r} box is pinned to {edges} image borders in {asset.original_path}",
                        asset.id,
                        asset.original_path,
                        class_id,
                    )
                )
    return findings


# --- dataset-level checks -----------------------------------------------------


def _audit_class_balance(class_counts: Counter[str], limits: AuditThresholds) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if not class_counts:
        return findings

    for class_id, count in sorted(class_counts.items()):
        if count < limits.min_class_count:
            findings.append(
                AuditFinding(
                    "class.rare",
                    f"Class {class_id!r} has only {count} labeled object(s) (threshold {limits.min_class_count}) — "
                    "metrics on it will be noise",
                    class_id=class_id,
                )
            )

    if len(class_counts) >= 2:
        most = class_counts.most_common()
        top_class, top_count = most[0]
        bottom_class, bottom_count = most[-1]
        if bottom_count > 0 and top_count / bottom_count > limits.imbalance_ratio:
            findings.append(
                AuditFinding(
                    "class.imbalance",
                    f"Class imbalance {top_count / bottom_count:.1f}:1 — {top_class!r} has {top_count} objects, "
                    f"{bottom_class!r} has {bottom_count} (threshold {limits.imbalance_ratio:.0f}:1)",
                    class_id=bottom_class,
                )
            )
    return findings
