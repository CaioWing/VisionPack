---
title: Cloud Sync (S3/GCS)
nav_order: 5
---

# Cloud Sync — S3 / GCS / Azure
{: .no_toc }

VisionPack syncs datasets that live in object stores **without downloading the
whole thing** and **without duplicating bytes** — while keeping the same
content-addressed integrity guarantees as local projects.

1. TOC
{:toc}

## Install a backend

Cloud backends are optional extras (they pull `fsspec` + the provider library):

```bash
pip install "visionpack[s3]"      # Amazon S3
pip install "visionpack[gcs]"     # Google Cloud Storage
pip install "visionpack[azure]"   # Azure Blob
```

## The mental model

- **Identity is always the `sha256` of the content** — the same as local. Each
  object is read **once** to hash it, then never re-read.
- **`etag` is a change-detector, never an identity.** "Same key + same etag" means
  the bytes are unchanged, which is all a re-sync needs. VisionPack never compares
  etags across different keys, so multipart etags, SSE-KMS, and crc32c quirks can't
  cause a mismatch.
- **Re-sync is metadata-only.** Re-running lists object metadata, sees the etags
  match, and does nothing — no downloads, no copies.

{: .note }
Transfers are **server-side within one provider** (S3↔S3, GCS↔GCS) — the bytes
never touch your machine. **Cross-provider** targets (S3→GCS, local→S3, S3→local
…) also work: sync *relays* the bytes it already read to compute the `sha256`,
so a cross-provider copy still costs exactly **one read + one upload**, never a
second download.

## Declare remote sources

Sources take remote URIs anywhere a local path would go. Credentials and region
are declared per source (never raw secrets in the manifest — point at a profile or
let ambient/instance-role auth apply):

```yaml
sources:
  - name: camera-A
    format: yolo
    images:
      uri: s3://my-bucket/camera-a/images
      region: us-east-1
    labels: s3://my-bucket/camera-a/labels
    classes: s3://my-bucket/camera-a/classes.txt
    match: relpath
    copy: copy            # see "Copy modes" below
```

Preview without writing anything — this lists metadata only, no object bodies:

```bash
vp sync --dry-run
```

```text
[camera-A] yolo
  images: s3://my-bucket/camera-a/images
  found 5,120 images, 5,031 labels -> 5,031 matched
  classes: scratch, dent, crack
```

Then reconcile. Re-running is idempotent — unchanged objects are skipped entirely:

```bash
vp sync
vp sync --source camera-A   # just one source
```

## A content-addressed target

Set a `target:` and `copy` mode lands objects in a self-sufficient,
content-addressed bucket — **server-side**, so the bytes never round-trip through
your machine:

```yaml
target: s3://my-bucket/datasets/factory-defects

sources:
  - name: camera-A
    format: yolo
    images: s3://my-bucket/raw/camera-a/images
    labels: s3://my-bucket/raw/camera-a/labels
    copy: copy
```

Objects land at `…/datasets/factory-defects/objects/sha256/<ab>/<cd>/<sha>`. The
same image arriving from any source dedups to the same key, copied at most once.

The `target:` can also carry `region`/`credentials`:

```yaml
target:
  uri: s3://my-bucket/datasets/factory-defects
  region: us-east-1
```

## Copy modes

Pick how each source materializes its bytes with `copy:`.

| Mode | What it does | Use when |
|------|--------------|----------|
| `copy` | Copy into the `target:` content-addressed store — server-side when source and target share a provider, single-pass relay when they don't. Target is self-sufficient; global dedup. | The common cloud case. |
| `reference` | No copy — the index points straight at the source object. | You control the source bucket and want zero extra storage. |
| `ingest` | Download into the **local** CAS (`.vp/objects/`). | Offline / edge work on a remote dataset. |

{: .warning }
There is no `move` mode — it was the only irreversible operation. To drain a
staging bucket, use `copy` plus a cloud lifecycle policy.

## Export for training (streaming)

A cloud-backed dataset doesn't get downloaded at export time either. `vp export`
writes the labels and a `manifest.jsonl` mapping each image to its object URI, so
a trainer streams images straight from the bucket:

```bash
vp export --format yolo --split
```

```text
exports/yolo/
  labels/{train,val,test}/
  classes.txt
  data.yaml
  manifest.jsonl        # {"image": "...", "uri": "s3://…", "label": "...", "set": "train"}
```

For **local** datasets the same command hardlinks images from the CAS — zero extra
bytes. You don't choose between the two; VisionPack picks per asset.

## What a re-sync actually costs

The first time an object is seen it's read once (streamed, not persisted) to
compute its `sha256`. After that, a re-sync is a metadata listing — etags match,
the delta is empty, nothing is read or copied. That's the whole recurring cost.

## See also

- [CLI Guide]({% link usage.md %}) — `vp sync` / `vp export` options.
- [Quickstart]({% link quickstart.md %}) — the local-first walkthrough.
