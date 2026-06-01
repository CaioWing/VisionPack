from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from visionpack.core.project import Project
from visionpack.media import image_info


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    asset_id: str | None = None
    path: str | None = None


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_project(project: Project, strict: bool = False) -> ValidationReport:
    issues: list[ValidationIssue] = []
    class_ids = project.manifest.class_ids
    assets = project.index.assets()
    annotations = project.index.annotations()
    asset_by_id = {asset.id: asset for asset in assets}
    annotations_by_asset = {annotation.asset_id: annotation for annotation in annotations}

    for asset in assets:
        source = asset.resolved_path(project.root)
        try:
            width, height, _, _ = image_info(source)
            if width != asset.width or height != asset.height:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "image.dimensions_changed",
                        f"Image dimensions changed for {asset.original_path}: indexed {asset.width}x{asset.height}, found {width}x{height}",
                        asset.id,
                        str(source),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - validation should aggregate all readable failures
            issues.append(
                ValidationIssue(
                    "error",
                    "image.unreadable",
                    f"Unreadable image for {asset.original_path}: {exc}",
                    asset.id,
                    str(source),
                )
            )

        if asset.id not in annotations_by_asset:
            severity = "error" if strict or project.manifest.validation.get("require_annotations") else "warning"
            issues.append(
                ValidationIssue(
                    severity,
                    "annotation.missing",
                    f"Image has no annotation: {asset.original_path}",
                    asset.id,
                    asset.original_path,
                )
            )

    for annotation in annotations:
        asset = asset_by_id.get(annotation.asset_id)
        if asset is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "annotation.orphan",
                    f"Annotation {annotation.id} references missing asset {annotation.asset_id}",
                    annotation.asset_id,
                )
            )
            continue
        for obj in annotation.objects:
            if class_ids and obj.class_id not in class_ids:
                issues.append(
                    ValidationIssue(
                        "error",
                        "class.unknown",
                        f"Unknown class {obj.class_id} in {asset.original_path}",
                        asset.id,
                        asset.original_path,
                    )
                )
            bbox = obj.bbox
            min_area = project.manifest.validation.get("bbox", {}).get("min_area_px", 0)
            if bbox.width <= 0 or bbox.height <= 0 or bbox.width * bbox.height < min_area:
                issues.append(
                    ValidationIssue(
                        "error",
                        "bbox.zero_area",
                        f"Invalid bounding box area in {asset.original_path}",
                        asset.id,
                        asset.original_path,
                    )
                )
            allow_oob = project.manifest.validation.get("bbox", {}).get("allow_out_of_bounds", False)
            if not allow_oob and (bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.width > asset.width or bbox.y + bbox.height > asset.height):
                issues.append(
                    ValidationIssue(
                        "error",
                        "bbox.out_of_bounds",
                        f"Bounding box exceeds image bounds in {asset.original_path}",
                        asset.id,
                        asset.original_path,
                    )
                )

    for path in project.index.orphan_labels():
        issues.append(ValidationIssue("warning", "label.orphan", f"Label file has no matching image: {path}", path=path))

    duplicate_hashes = [sha for sha, count in Counter(asset.sha256 for asset in assets).items() if count > 1]
    for sha in duplicate_hashes:
        issues.append(ValidationIssue("warning", "asset.duplicate_exact", f"Duplicate asset content detected: sha256:{sha}"))

    for split in project.index.splits():
        seen: dict[str, str] = {}
        for split_name, asset_ids in split.sets.items():
            for asset_id in asset_ids:
                if asset_id in seen:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "split.leakage",
                            f"Asset {asset_id} appears in both {seen[asset_id]} and {split_name} for split {split.id}",
                            asset_id,
                        )
                    )
                seen[asset_id] = split_name
                if asset_id not in asset_by_id:
                    issues.append(
                        ValidationIssue("error", "split.missing_asset", f"Split {split.id} references missing asset {asset_id}", asset_id)
                    )

    return ValidationReport(issues)
