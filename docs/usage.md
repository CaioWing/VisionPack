# VisionPack Usage Guide

VisionPack is a CLI-first DatasetOps tool for computer vision datasets. It supports
local projects across classification, detection, instance segmentation, and
keypoints; YOLO / COCO / ImageFolder import and export; declarative multi-source
sync; validation (including near-duplicate and cross-split leakage detection);
statistics; deterministic splits; snapshots and snapshot diffs; and archive and
WebDataset training packs.

For architecture and the roadmap see [ARCHITECTURE.md](../ARCHITECTURE.md).

## Install For Development

From the repository root:

```bash
uv sync
uv run vp --help
```

You can also run the module directly:

```bash
uv run python -m visionpack --help
```

## Initialize A Dataset

Create a VisionPack project in the current directory:

```bash
uv run vp init --name factory-defects --task detection
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
uv run vp import ./raw --format yolo
```

By default, VisionPack ingests images into `.vp/objects/sha256` and indexes them by content hash. You can choose another copy mode:

```bash
uv run vp import ./raw --format yolo --copy hardlink
uv run vp import ./raw --format yolo --copy reference
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
uv run vp sync --dry-run   # preview found / matched / unmatched / classes per source
uv run vp sync             # ingest; idempotent, records per-asset provenance
uv run vp sync --source camera-A   # sync just one source
```

## Validate

Run validation:

```bash
uv run vp validate
```

Strict mode treats missing annotations as errors:

```bash
uv run vp validate --strict
```

Write a JSON validation report:

```bash
uv run vp validate --report reports/validation.json
```

The current validator checks image readability, missing annotations, orphan labels, unknown classes, invalid boxes, boxes outside image bounds, duplicate exact assets, and split leakage.

## Show Statistics

Print a summary:

```bash
uv run vp stats
```

Class distribution:

```bash
uv run vp stats --by class
```

JSON output:

```bash
uv run vp stats --json
```

## Create Snapshots

Create a reproducible dataset snapshot:

```bash
uv run vp snapshot create -m "initial import"
```

List snapshots:

```bash
uv run vp snapshot list
```

Show one snapshot:

```bash
uv run vp snapshot show v1
```

Snapshots store hashes for the manifest, assets, annotations, splits, and summary stats. They are written to `.vp/snapshots/`.

## Diff Snapshots

Compare two snapshots:

```bash
uv run vp diff v1 v2
```

JSON diff:

```bash
uv run vp diff v1 v2 --json
```

The diff reports added and removed assets, added/removed/modified annotations, class changes, split changes, and before/after stats.

## Export YOLO

Export the indexed dataset back to YOLO format:

```bash
uv run vp export --format yolo --output exports/yolo-v1
```

The export writes:

```text
exports/yolo-v1/
  images/
  labels/
  classes.txt
  data.yaml
```

## Pack Archive

Create a compressed archive package:

```bash
uv run vp pack --profile archive
```

Or choose an output path:

```bash
uv run vp pack --profile archive --output exports/archive/factory-defects.tar.zst
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
- remote source backends (S3/GCS/Azure/git) are planned; sources are local for now
- `vp eval` (scoring predictions into benchmark metrics) is planned
- the local index is SQLite (`index.db`); full-scan commands still materialize all
  records into RAM (streaming reads are the next scale step — see
  [ARCHITECTURE.md](../ARCHITECTURE.md))
