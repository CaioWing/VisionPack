---
title: Home
nav_order: 1
description: "VisionPack — DatasetOps for computer vision. Version, validate, split, and export image datasets reproducibly, from your laptop to S3/GCS."
permalink: /
---

# VisionPack
{: .fs-9 }

DatasetOps for computer vision — a Git/Docker-like CLI for the messy part of
training models: turning scattered images and labels into a clean, **versioned,
leak-free, ready-to-train** dataset.
{: .fs-6 .fw-300 }

[Get started]({% link quickstart.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Install]({% link installation.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/CaioWing/VisionPack){: .btn .fs-5 .mb-4 .mb-md-0 }

---

```bash
pip install visionpack
```

```bash
vp init --name factory-defects --task detection
vp sync                          # pull images + labels from visionpack.yaml sources
vp validate                      # catch corrupt images, bad boxes, near-dup leakage
vp split create --strategy stratified
vp snapshot create -m "baseline"
vp export --format yolo --split  # ready-to-train layout
```

## Why VisionPack

Most of the pain in a CV project isn't the model — it's the dataset. VisionPack
targets the failures that quietly cost you accuracy and reproducibility:

- **Train/test leakage that inflates your metrics.** Exact-duplicate detection
  isn't enough: a re-encoded or resized copy of a training image landing in the
  test set makes your reported numbers a lie. VisionPack catches **near-duplicate**
  leakage with perceptual hashing.
- **Splits you can't reproduce.** "I shuffled with `seed=42`" breaks the moment
  data is added or reordered. VisionPack splits are a function of image *content*,
  so they're identical across machines and stable as the dataset grows.
- **Data scattered across places.** Images in one bucket, labels in another repo,
  classes in a third. Declare them once and `vp sync` assembles the dataset —
  locally or straight from **S3 / GCS / Azure**.
- **"Which dataset trained this model?"** Content-addressed snapshots make that
  answerable instead of a guess.

It complements — not replaces — CVAT, FiftyOne, DVC, Roboflow, and Label Studio.
VisionPack is the DatasetOps layer that imports, validates, versions, splits,
packs, and exports.

## Works across the common CV tasks

| Task | Import | Geometry |
|------|--------|----------|
| **Classification** | ImageFolder (folder-per-class) | whole-image label |
| **Detection** | YOLO, COCO | bounding box |
| **Instance segmentation** | YOLO-seg, COCO | polygon |
| **Semantic segmentation (export)** | — | class-index mask PNGs (`--format masks`) |
| **Keypoints / pose** | COCO | keypoints |

## What's in the box

- **Deterministic, lockable splits** — `stratified` / `random` / `hash`, captured in snapshots.
- **Near-duplicate & cross-split leakage detection** — perceptual-hash tier, scale-proof via LSH bucketing.
- **Multi-source sync** — declarative `sources:` + `vp sync`, with per-asset provenance.
- **Cloud-native, multi-provider** — sync YOLO/COCO/ImageFolder sources from and to
  S3/GCS/Azure without downloading the whole dataset, including cross-provider
  targets; see [Cloud Sync]({% link cloud-sync.md %}).
- **Content-addressed snapshots & diff** — reproducible versions; compare any two.
- **Strong validation** — unreadable images, missing/orphan labels, unknown classes, bad boxes, exact + near duplicates, split leakage.
- **Benchmarking & model-in-the-loop** — `vp eval` (mAP / accuracy on a locked split),
  `vp autolabel` (confident predictions become labels), `vp queue` (what to label next).
- **Packing & export** — `archive` (`.tar.zst`) and `training` (WebDataset shards); byte-free exports via hardlinks / streaming manifests.
- **Machine-readable output** — every pipeline command takes `--json` and prints a
  stable, schema-versioned envelope; see [JSON Output]({% link json-output.md %}).

## Next steps

- [Install VisionPack]({% link installation.md %}) — `pip install visionpack`
- [Quickstart]({% link quickstart.md %}) — a dataset in 60 seconds
- [CLI Guide]({% link usage.md %}) — every command and its options
- [Cloud Sync]({% link cloud-sync.md %}) — S3 / GCS / Azure datasets
- [JSON Output]({% link json-output.md %}) — drive VisionPack from other programs

{: .note }
VisionPack is in **active development** (early but usable). The end-to-end
workflow works across classification, detection, instance segmentation, and
keypoints. APIs may still shift — feedback and contributions are welcome.
