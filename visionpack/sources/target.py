from __future__ import annotations

from dataclasses import dataclass

from visionpack.sources.resolver import Resolver


@dataclass(slots=True)
class CloudTarget:
    """A content-addressed sink objects are copied into, server-side.

    Mirrors the local CAS layout (``objects/sha256/<ab>/<cd>/<sha>``) in a target
    bucket so the target is self-sufficient and dedups globally by content: the
    same image arriving from any source lands on the same key, copied at most
    once. See docs/SPEC-cloud-sync.md.
    """

    base_uri: str
    resolver: Resolver

    def object_uri(self, sha256: str) -> str:
        return f"{self.base_uri.rstrip('/')}/objects/sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"

    def ensure_object(self, src_uri: str, sha256: str) -> str:
        """Server-copy ``src_uri`` into its content-addressed slot, returning it.

        Idempotent: if the slot already holds the object (this run, a prior run,
        or another machine), the copy is skipped — content addressing makes the
        ``exists`` check a safe dedup, since identical content means identical key.
        """
        destination = self.object_uri(sha256)
        if not self.resolver.exists(destination):
            self.resolver.server_copy(src_uri, destination)
        return destination
