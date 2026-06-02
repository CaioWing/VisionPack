from __future__ import annotations

import hashlib
import struct
from collections import Counter, defaultdict

from visionpack.core.errors import VisionPackError
from visionpack.core.models import Asset, Split
from visionpack.core.project import Project

# Assets with no annotations are grouped here for stratification purposes.
BACKGROUND = "__background__"

_SET_ORDER = ("train", "val", "test")


def create_split(
    project: Project,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    strategy: str = "stratified",
    seed: int = 0,
    split_id: str = "default",
    by: str = "class",
    force: bool = False,
) -> Split:
    """Create a deterministic, versionable train/val/test split.

    Determinism comes from hashing each asset's *content* hash with the seed, so
    the same dataset and seed always produce the same split, independent of
    import order or machine. Exact-duplicate images share one asset record (the
    id derives from the content hash), so they can never straddle two sets.

    Strategies:

    - ``stratified`` (default): balance each class's images across the sets,
      best for getting comparable, low-variance metrics across rare classes.
    - ``random``: a single deterministic shuffle cut at exact global ratios.
    - ``hash``: assign each asset by a threshold on its own hash. The only
      strategy that is *stable as the dataset grows* — adding new images never
      reassigns existing ones — at the cost of approximate (expected) ratios.
    """
    ratios = {name: float(value) for name, value in (("train", train), ("val", val), ("test", test)) if value and value > 0}
    if not ratios:
        raise VisionPackError("At least one of --train/--val/--test must be greater than 0.")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise VisionPackError(f"Split ratios must sum to 1.0 (got {total:.4f} from {ratios}).")
    if strategy not in {"stratified", "random", "hash"}:
        raise VisionPackError(f"Unknown split strategy {strategy!r}. Use one of: stratified, random, hash.")
    if by != "class":
        raise VisionPackError(f"Unsupported stratification key {by!r}. Only --by class is supported.")

    existing = {item.id: item for item in project.index.splits()}
    if split_id in existing and existing[split_id].locked and not force:
        raise VisionPackError(f"Split {split_id!r} is locked. Pass --force to overwrite it.")

    assets = project.index.assets()
    ordered_names = [name for name in _SET_ORDER if name in ratios]

    if strategy == "hash":
        sets = _assign_by_threshold(assets, ratios, ordered_names, seed)
    elif strategy == "random":
        sets = _assign_by_rank(_hash_sorted_ids(assets, seed), ratios, ordered_names)
    else:
        sets = {name: [] for name in ordered_names}
        groups = _group_by_primary_class(project, assets)
        for _, group_assets in sorted(groups.items()):
            group_sets = _assign_by_rank(_hash_sorted_ids(group_assets, seed), ratios, ordered_names)
            for name in ordered_names:
                sets[name].extend(group_sets[name])

    sets = {name: sorted(ids) for name, ids in sets.items()}
    split = Split(id=split_id, strategy=strategy, sets=sets, locked=False)
    project.index.upsert_split(split)
    project.index.save()
    return split


def get_split(project: Project, split_id: str = "default") -> Split | None:
    return next((item for item in project.index.splits() if item.id == split_id), None)


def asset_set_map(split: Split) -> dict[str, str]:
    """Map each asset id to the name of the set it belongs to."""
    return {asset_id: set_name for set_name, asset_ids in split.sets.items() for asset_id in asset_ids}


def resolve_export_sets(project: Project, split_id: str | None):
    """Resolve how an export should partition assets.

    Returns ``(set_for_asset, ordered_set_names)`` where ``set_for_asset(asset_id)``
    gives the set an asset belongs to (or ``None`` to skip it). With ``split_id``
    of ``None`` every asset maps to a single bucket, so callers can treat flat and
    split exports uniformly. Set names are ordered train, val, test first.
    """
    if split_id is None:
        return (lambda asset_id: "all"), []
    split = get_split(project, split_id)
    if split is None:
        raise VisionPackError(f"No split named {split_id!r}. Create one with `vp split create`.")
    membership = asset_set_map(split)
    ordered = [name for name in _SET_ORDER if name in split.sets]
    ordered += [name for name in split.sets if name not in _SET_ORDER]
    return (lambda asset_id: membership.get(asset_id)), ordered


def lock_split(project: Project, split_id: str = "default") -> Split:
    existing = {item.id: item for item in project.index.splits()}
    split = existing.get(split_id)
    if split is None:
        raise VisionPackError(f"No split named {split_id!r}. Create one with `vp split create`.")
    split.locked = True
    project.index.upsert_split(split)
    project.index.save()
    return split


def _uniform01(seed: int, sha256: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sha256}".encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0] / float(1 << 64)


def _hash_sorted_ids(assets: list[Asset], seed: int) -> list[str]:
    # Order is a function of content only, so it never depends on import order.
    return [asset.id for asset in sorted(assets, key=lambda item: (_uniform01(seed, item.sha256), item.sha256))]


def _assign_by_threshold(assets: list[Asset], ratios: dict[str, float], ordered_names: list[str], seed: int) -> dict[str, list[str]]:
    thresholds: list[tuple[str, float]] = []
    acc = 0.0
    for name in ordered_names:
        acc += ratios[name]
        thresholds.append((name, acc))
    sets: dict[str, list[str]] = {name: [] for name in ordered_names}
    for asset in assets:
        value = _uniform01(seed, asset.sha256)
        chosen = ordered_names[-1]
        for name, threshold in thresholds:
            if value < threshold:
                chosen = name
                break
        sets[chosen].append(asset.id)
    return sets


def _assign_by_rank(asset_ids: list[str], ratios: dict[str, float], ordered_names: list[str]) -> dict[str, list[str]]:
    counts = _largest_remainder([ratios[name] for name in ordered_names], len(asset_ids))
    sets: dict[str, list[str]] = {}
    index = 0
    for name, count in zip(ordered_names, counts):
        sets[name] = asset_ids[index : index + count]
        index += count
    return sets


def _largest_remainder(ratios: list[float], n: int) -> list[int]:
    """Apportion ``n`` items across ratios so the counts sum exactly to ``n``."""
    raw = [ratio * n for ratio in ratios]
    floors = [int(value) for value in raw]
    remainder = n - sum(floors)
    by_fraction = sorted(range(len(ratios)), key=lambda i: (raw[i] - floors[i], ratios[i]), reverse=True)
    for i in range(remainder):
        floors[by_fraction[i]] += 1
    return floors


def _primary_class(project: Project, asset: Asset) -> str:
    annotation = project.index.annotation_for_asset(asset.id)
    if annotation is None or not annotation.objects:
        return BACKGROUND
    counts = Counter(obj.class_id for obj in annotation.objects)
    # Most frequent class; ties broken by class id for determinism.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _group_by_primary_class(project: Project, assets: list[Asset]) -> dict[str, list[Asset]]:
    groups: dict[str, list[Asset]] = defaultdict(list)
    for asset in assets:
        groups[_primary_class(project, asset)].append(asset)
    return groups
