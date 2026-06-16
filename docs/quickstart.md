---
title: Quickstart
nav_order: 3
---

# Quickstart — a dataset in 60 seconds
{: .no_toc }

This walkthrough takes a raw YOLO dataset to a validated, versioned,
ready-to-train export. Every step is idempotent and reproducible.

1. TOC
{:toc}

## 0. Install

```bash
pip install visionpack
```

## 1. Create a project

```bash
vp init --name factory-defects --task detection
```

This writes a git-like layout — just the manifest and a control directory:

```text
visionpack.yaml      # the declarative dataset manifest (source of truth)
.vp/
  db/                # local index (index.db, SQLite)
  objects/           # content-addressed assets (sha256)
  snapshots/         # versioned snapshots
```

## 2. Bring in some data

Point VisionPack at a YOLO dataset. It hashes each image, stores it once by
content, and pairs it with its label:

```bash
vp import ./raw --format yolo
```

```text
Imported 1,284 images, 1,190 labels, 4,532 objects
3 classes merged into visionpack.yaml
94 images without a matching label
```

{: .note }
A one-off `import` is also recorded as a source in `visionpack.yaml`, so the
manifest stays the single source of truth and you can re-pull later with
`vp sync`. Pass `--no-record` for a throwaway import.

## 3. Validate for real problems

```bash
vp validate
```

```text
✓ 1,284 images readable
✗ 2 boxes outside image bounds  (asset_9f2a…, asset_b1c4…)
⚠ 7 near-duplicate pairs (perceptual)  — 1 crosses train/test
```

Validation covers unreadable images, missing/orphan labels, unknown classes,
invalid and out-of-bounds boxes, exact + **near-duplicate** images, and
cross-split leakage. Use `--strict` to fail on missing annotations, or
`--report reports/validation.json` for machine-readable output.

## 4. A deterministic, reproducible split

```bash
vp split create --train 0.8 --val 0.1 --test 0.1 --strategy stratified
vp split lock
```

Splits are a function of image **content**, not a random seed — identical across
machines and stable as the dataset grows. `lock` freezes the assignment so later
runs can't silently reshuffle it.

## 5. Freeze a reproducible version

```bash
vp snapshot create -m "initial import"
```

Content-addressed snapshots answer "which dataset trained this model?" — compare
any two with `vp diff v1 v2`.

## 6. Check class balance

```bash
vp stats --by split
```

## 7. Export a ready-to-train layout

```bash
vp export --format yolo --split
```

```text
exports/yolo/
  images/{train,val,test}/
  labels/{train,val,test}/
  classes.txt
  data.yaml
```

Local exports **hardlink** from the content-addressed store, so they cost zero
extra bytes.

## Where to go next

- [CLI Guide]({% link usage.md %}) — every command and option.
- [Cloud Sync]({% link cloud-sync.md %}) — do all of this against S3 / GCS / Azure.
