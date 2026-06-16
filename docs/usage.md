---
title: CLI Guide
nav_order: 4
---

# VisionPack CLI Guide

VisionPack is a CLI-first DatasetOps tool for computer vision datasets. It supports
local projects across classification, detection, instance segmentation, and
keypoints; YOLO / COCO / ImageFolder import and export; declarative multi-source
sync; validation (including near-duplicate and cross-split leakage detection);
statistics; deterministic splits; snapshots and snapshot diffs; and archive and
WebDataset training packs.

For architecture and the roadmap see
[ARCHITECTURE.md](https://github.com/CaioWing/VisionPack/blob/main/ARCHITECTURE.md).

## Install

```bash
pip install visionpack
```

Cloud backends are optional extras: `pip install "visionpack[s3]"` (also `[gcs]`,
`[azure]`). See [Installation]({% link installation.md %}) for details.

For development from source with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
uv run vp --help
uv run python -m visionpack --help   # or run the module directly
```

The examples below use the bare `vp` command; prefix with `uv run` when working
from a source checkout.

## Initialize A Dataset

Create a VisionPack project in the current directory:

```bash
vp init --name factory-defects --task detection
```

This creates a git-like layout — just the manifest and a control directory:

```text
visionpack.yaml
.vp/
  db/          # local index (index.db, SQLite)
  objects/     # content-addressed assets (sha256)
  snapshots/   # versioned snapshots
```

The `visionpack.yaml` file is the declarative dataset manifest and, together with
the `.vp/` index, is the single source of truth. Output directories such as
`exports/` and `reports/` are created on demand by the commands that write them, so
the project root stays clean.

## Import A YOLO Dataset

Supported YOLO layouts include image and label files side by side:

```text
raw/
  classes.txt
  img001.jpg
  img001.txt
```

And split-style image/label folders:

```text
raw/
  classes.txt
  images/
    img001.jpg
  labels/
    img001.txt
```

Import the dataset:

```bash
vp import ./raw --format yolo
```

By default, VisionPack ingests images into `.vp/objects/sha256` and indexes them by content hash. You can choose another copy mode:

```bash
vp import ./raw --format yolo --copy hardlink
vp import ./raw --format yolo --copy reference
```

Available copy modes are `copy`, `move`, `hardlink`, `reference`, and `ingest`.

Each successful import is also recorded as a source in `visionpack.yaml`, so the
manifest reflects where your data came from and you can re-pull it later with
`vp sync`. Pass `--no-record` to skip this (for a one-off/throwaway import), or
`--name` to control the recorded source name.

## Multi-source sync

Declare images and labels — even when they live in different folders or repos — in a
`sources:` block and reconcile them with `vp sync`:

```yaml
sources:
  - name: camera-A
    format: yolo
    images: ./repoA/images
    labels: ./repoB
    match: stem          # pair by filename; use `relpath` for parallel trees
    copy: ingest
```

```bash
vp sync --dry-run   # preview found / matched / unmatched / classes per source
vp sync             # ingest; idempotent, records per-asset provenance
vp sync --source camera-A   # sync just one source
```

Sources can also live in object stores. Remote URIs go anywhere a local path
would, and a `target:` lets `copy` mode land objects server-side in a
content-addressed bucket without downloading them:

```yaml
target: s3://my-bucket/datasets/factory-defects
sources:
  - name: camera-A
    format: yolo
    images: s3://my-bucket/raw/camera-a/images
    labels: s3://my-bucket/raw/camera-a/labels
    copy: copy
```

See [Cloud Sync]({% link cloud-sync.md %}) for credentials, copy modes, and
streaming export.

## Validate

Run validation:

```bash
vp validate
```

Strict mode treats missing annotations as errors:

```bash
vp validate --strict
```

Write a JSON validation report:

```bash
vp validate --report reports/validation.json
```

The current validator checks image readability, missing annotations, orphan labels, unknown classes, invalid boxes, boxes outside image bounds, duplicate exact assets, and split leakage.

## Show Statistics

Print a summary:

```bash
vp stats
```

Class distribution:

```bash
vp stats --by class
```

JSON output:

```bash
vp stats --json
```

## Create Snapshots

Create a reproducible dataset snapshot:

```bash
vp snapshot create -m "initial import"
```

List snapshots:

```bash
vp snapshot list
```

Show one snapshot:

```bash
vp snapshot show v1
```

Snapshots store hashes for the manifest, assets, annotations, splits, and summary stats. They are written to `.vp/snapshots/`.

## Diff Snapshots

Compare two snapshots:

```bash
vp diff v1 v2
```

JSON diff:

```bash
vp diff v1 v2 --json
```

The diff reports added and removed assets, added/removed/modified annotations, class changes, split changes, and before/after stats.

## Export YOLO

Export the indexed dataset back to YOLO format:

```bash
vp export --format yolo --output exports/yolo-v1
```

The export writes:

```text
exports/yolo-v1/
  images/
  labels/
  classes.txt
  data.yaml
```

Exports never duplicate bytes: **local** images are hardlinked from the
content-addressed store, and **cloud-backed** images are written to a
`manifest.jsonl` (image → object URI) for streaming instead of being downloaded.
See [Cloud Sync]({% link cloud-sync.md %}#export-for-training-streaming).

## Pack Archive

Create a compressed archive package:

```bash
vp pack --profile archive
```

Or choose an output path:

```bash
vp pack --profile archive --output exports/archive/factory-defects.tar.zst
```

The archive includes:

- `visionpack.yaml`
- `.vp/db/index.db`
- `.vp/snapshots/*.json`
- content-addressed assets
- `pack.json` with pack metadata and dataset stats

## Python API

The public API is intentionally small while the project is early:

```python
from visionpack import Dataset

ds = Dataset.open(".")
print(ds.manifest.name)
print(len(ds.index.assets()))
```

More stable SDK methods will be added as the internal workflows settle.

## Current Limitations

- semantic segmentation (per-class mask PNGs) is not yet supported; instance
  segmentation (polygons) and keypoints are supported via COCO
- YOLO import/export is detection-only (no YOLO-seg / YOLO-pose yet)
- `--format auto` detection is not implemented; pass `--format` explicitly
- `vp annotate` is scaffolded but not implemented yet
- cloud sync (S3/GCS/Azure) is **same-provider** in v1 — cross-cloud transfer
  (S3↔GCS) and remote COCO/ImageFolder sync are planned; `pack` is local-only
- `vp eval` (scoring predictions into benchmark metrics) is planned
- the local index is SQLite (`index.db`); `stats` and exports stream records (flat
  RAM at scale), while `validate`, `split`, dedup, and the WebDataset pack still load
  the full set (next streaming pass — see
  [ARCHITECTURE.md](https://github.com/CaioWing/VisionPack/blob/main/ARCHITECTURE.md))
