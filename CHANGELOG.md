# Changelog

All notable changes to VisionPack are tracked here.

## [Unreleased]

- Cloud sync foundation (PR 1 of docs/SPEC-cloud-sync.md): `FsspecResolver` for
  remote schemes (`s3://`, `gs://`, `az://`) behind optional extras
  (`visionpack[s3]`, `[gcs]`, `[azure]`); `Resolver.stat` for metadata-only
  probing; and a `blob_cache` so an unchanged re-sync skips re-reading object
  bodies entirely. Listings carry size+etag in one pass (no per-object HEAD) and
  credentials/region from the YAML reach the provider filesystem.
- Cloud-internal sync (PR 2): a `target:` in `visionpack.yaml` is a
  content-addressed sink that `copy`-mode sync lands objects in **server-side**
  (`Resolver.server_copy`), so bytes never round-trip through the client; global
  dedup by content, idempotent re-sync, and non-destructive (no `move`).
  `reference` mode points the index straight at the source object. Same-provider
  only in v1.
- Byte-free export (PR 3): `vp export` (yolo/coco/imagefolder) now **hardlinks**
  local images from the CAS instead of copying them — exports cost zero extra
  bytes. Cloud-backed (remote) assets are not downloaded: each is written to a
  `manifest.jsonl` of `(image, uri, ...)` alongside the generated labels, so a
  trainer streams images straight from the bucket.

## [0.0.1] - 2026-06-15

- No changes documented.

## [0.1.0] - 2026-06-04

- Initial PyPI-ready package metadata, CLI entry points, build workflow, and release publishing workflow.
