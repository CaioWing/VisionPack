from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from visionpack.core.errors import VisionPackError
from visionpack.core.models import Asset
from visionpack.core.project import Project
from visionpack.perceptual import band_keys, dhash_path, hamming
from visionpack.split import asset_set_map, get_split

# Hamming distance (out of 64 bits) under which two images count as near
# duplicates. 0 means perceptually identical (e.g. a re-encoded JPEG); small
# values catch crops/resizes/watermarks. Conservative by default to avoid noise.
DEFAULT_THRESHOLD = 5


@dataclass(slots=True)
class NearDuplicatePair:
    asset_a: str
    asset_b: str
    distance: int


@dataclass(slots=True)
class NearDuplicateCluster:
    asset_ids: list[str]


@dataclass(slots=True)
class LeakagePair:
    asset_a: str
    asset_b: str
    set_a: str
    set_b: str
    distance: int


def phash_map(project: Project, assets: list[Asset] | None = None, *, persist: bool = False) -> dict[str, str]:
    """Return ``asset_id -> perceptual hash``.

    Uses the hash stored at import when present; otherwise computes it from the
    stored object so datasets imported before perceptual hashing existed are
    still covered. With ``persist`` the freshly computed hashes are written back
    to the index (a one-time backfill). Unreadable assets are skipped.
    """
    assets = assets if assets is not None else project.index.assets()
    result: dict[str, str] = {}
    backfilled: list[Asset] = []
    for asset in assets:
        if asset.phash:
            result[asset.id] = asset.phash
            continue
        try:
            value = dhash_path(asset.resolved_path(project.root))
        except VisionPackError:
            continue
        result[asset.id] = value
        asset.phash = value
        backfilled.append(asset)
    if persist and backfilled:
        for asset in backfilled:
            project.index.upsert_asset(asset)
        project.index.save()
    return result


def near_duplicate_pairs(phashes: dict[str, str], threshold: int = DEFAULT_THRESHOLD) -> list[NearDuplicatePair]:
    """Find every pair of assets within ``threshold`` bits, via LSH bucketing."""
    bands = max(1, threshold + 1)
    buckets: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for asset_id, phash in phashes.items():
        value = int(phash, 16)
        for key in band_keys(value, bands):
            buckets[key].append((asset_id, value))

    seen: set[tuple[str, str]] = set()
    pairs: list[NearDuplicatePair] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            a_id, a_val = bucket[i]
            for j in range(i + 1, len(bucket)):
                b_id, b_val = bucket[j]
                key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                if key in seen:
                    continue
                distance = (a_val ^ b_val).bit_count()
                if distance <= threshold:
                    seen.add(key)
                    pairs.append(NearDuplicatePair(key[0], key[1], distance))
    pairs.sort(key=lambda pair: (pair.distance, pair.asset_a, pair.asset_b))
    return pairs


def cluster_pairs(pairs: list[NearDuplicatePair]) -> list[NearDuplicateCluster]:
    """Group near-duplicate pairs into connected clusters (union-find)."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for pair in pairs:
        parent[find(pair.asset_a)] = find(pair.asset_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    clusters = [NearDuplicateCluster(sorted(ids)) for ids in groups.values() if len(ids) > 1]
    clusters.sort(key=lambda cluster: (-len(cluster.asset_ids), cluster.asset_ids[0]))
    return clusters


def leakage_from_pairs(pairs: list[NearDuplicatePair], membership: dict[str, str]) -> list[LeakagePair]:
    """Near-duplicate pairs whose two assets sit in different sets of a split."""
    leaks: list[LeakagePair] = []
    for pair in pairs:
        set_a = membership.get(pair.asset_a)
        set_b = membership.get(pair.asset_b)
        if set_a and set_b and set_a != set_b:
            leaks.append(LeakagePair(pair.asset_a, pair.asset_b, set_a, set_b, pair.distance))
    leaks.sort(key=lambda leak: (leak.distance, leak.asset_a, leak.asset_b))
    return leaks


def cross_split_leakage(project: Project, split_id: str = "default", threshold: int = DEFAULT_THRESHOLD) -> list[LeakagePair]:
    """High-level helper: near-duplicate images split across train/val/test.

    This is the metric-corrupting case — a test image that is a near copy of a
    training image inflates reported accuracy without anyone noticing.
    """
    split = get_split(project, split_id)
    if split is None:
        raise VisionPackError(f"No split named {split_id!r}. Create one with `vp split create`.")
    membership = asset_set_map(split)
    phashes = {asset_id: value for asset_id, value in phash_map(project).items() if asset_id in membership}
    pairs = near_duplicate_pairs(phashes, threshold)
    return leakage_from_pairs(pairs, membership)
