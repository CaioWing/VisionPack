from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from visionpack.core.errors import VisionPackError
from visionpack.core.models import Asset
from visionpack.core.project import Project
from visionpack.perceptual import band_keys, dhash_path
from visionpack.split import asset_set_map, get_split

# Hamming distance (out of 64 bits) under which two images count as near
# duplicates. 0 means perceptually identical (e.g. a re-encoded JPEG); small
# values catch crops/resizes/watermarks. Conservative by default to avoid noise.
DEFAULT_THRESHOLD = 5

# Full pairwise expansion of a duplicate group is quadratic: a dataset with
# thousands of visually identical frames (idle camera, flat backgrounds,
# calibration plates) would materialize hundreds of millions of pairs and run
# out of memory. Groups up to this size expand to every pair; larger groups
# pair each member with the group's first member instead — the connected
# clusters are identical, every asset still appears in at least one pair (so
# per-asset leakage stays visible), and memory stays linear.
GROUP_EXPANSION_CAP = 50


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
    missing: list[Asset] = []
    for asset in assets:
        if asset.phash:
            result[asset.id] = asset.phash
        else:
            missing.append(asset)

    backfilled: list[Asset] = []
    if missing:
        # Decoding image headers is I/O-bound and per-asset, so the backfill
        # fans out across threads (unreadable/remote assets come back as None).
        def compute(asset: Asset) -> str | None:
            try:
                return dhash_path(asset.resolved_path(project.root))
            except VisionPackError:
                return None

        with ThreadPoolExecutor() as pool:
            for asset, value in zip(missing, pool.map(compute, missing), strict=True):
                if value is None:
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
    """Find every pair of assets within ``threshold`` bits, via LSH bucketing.

    Assets sharing the *exact* hash are collapsed to one value first: batches of
    re-encoded near-uniform images (flat backgrounds, calibration plates) all
    land on the same hash, and running them through the bucket loop individually
    is the pathological O(n^2) case. The LSH comparison count is therefore
    bounded by the number of *distinct* hashes, and identical-hash groups expand
    to distance-0 pairs directly. Groups larger than ``GROUP_EXPANSION_CAP`` are
    star-expanded through their first member instead of enumerating every pair,
    keeping memory linear on datasets full of visually identical frames.
    """
    by_value: dict[int, list[str]] = defaultdict(list)
    for asset_id, phash in phashes.items():
        by_value[int(phash, 16)].append(asset_id)

    pairs: list[NearDuplicatePair] = []
    for ids in by_value.values():
        if len(ids) > 1:
            ids.sort()
            if len(ids) <= GROUP_EXPANSION_CAP:
                pairs.extend(NearDuplicatePair(a, b, 0) for i, a in enumerate(ids) for b in ids[i + 1 :])
            else:
                hub = ids[0]
                pairs.extend(NearDuplicatePair(hub, other, 0) for other in ids[1:])

    bands = max(1, threshold + 1)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for value in by_value:
        for key in band_keys(value, bands):
            buckets[key].append(value)

    # A pair over the threshold can never come back under it, so it is marked
    # seen too — no bucket sharing another band re-XORs the same pair.
    seen: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            a_val = bucket[i]
            for j in range(i + 1, len(bucket)):
                b_val = bucket[j]
                key = (a_val, b_val) if a_val < b_val else (b_val, a_val)
                if key in seen:
                    continue
                seen.add(key)
                distance = (a_val ^ b_val).bit_count()
                if distance <= threshold:
                    ids_a, ids_b = by_value[a_val], by_value[b_val]
                    if len(ids_a) * len(ids_b) <= GROUP_EXPANSION_CAP * GROUP_EXPANSION_CAP:
                        expansion = [(a_id, b_id) for a_id in ids_a for b_id in ids_b]
                    else:
                        # Star expansion across two huge groups: every member
                        # pairs with the other side's first member, so each
                        # asset still shows up without the full cross product.
                        hub_a, hub_b = ids_a[0], ids_b[0]
                        expansion = [(a_id, hub_b) for a_id in ids_a]
                        expansion += [(hub_a, b_id) for b_id in ids_b if b_id != hub_b]
                    for a_id, b_id in expansion:
                        first, second = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                        pairs.append(NearDuplicatePair(first, second, distance))
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
