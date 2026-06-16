---
title: Internals
nav_order: 8
has_children: true
---

# Internals

Design notes and specifications for how VisionPack works under the hood. Useful if
you're contributing, integrating, or just want to understand the guarantees.

- **[Core Spec]({% link SPEC.md %})** — the data model, storage, and command
  semantics.
- **[Cloud Sync Spec]({% link SPEC-cloud-sync.md %})** — how remote sync stays
  efficient and safe (sha256 identity, etag-as-change-detector, server-side copy).

For the broader architecture, module map, and roadmap, see
[ARCHITECTURE.md](https://github.com/CaioWing/VisionPack/blob/main/ARCHITECTURE.md)
in the repository.
