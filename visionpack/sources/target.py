from __future__ import annotations

from dataclasses import dataclass

from visionpack.core.errors import VisionPackError
from visionpack.sources.resolver import Resolver


@dataclass(slots=True)
class CloudTarget:
    """A content-addressed sink objects are copied into.

    Mirrors the local CAS layout (``objects/sha256/<ab>/<cd>/<sha>``) in a target
    bucket so the target is self-sufficient and dedups globally by content: the
    same image arriving from any source lands on the same key, copied at most
    once. See docs/SPEC-cloud-sync.md.

    ``server_side`` says whether source and target live on the same provider.
    When they do, objects move with a server-side copy (S3 ``CopyObject`` / GCS
    rewrite) and never transit the client. When they don't (cross-provider:
    S3 -> GCS, local -> S3, ...), the sync *relays* the bytes it already read
    for hashing — a single read (needed anyway for the sha256) plus one upload,
    never a second download.
    """

    base_uri: str
    resolver: Resolver
    server_side: bool = True

    def object_uri(self, sha256: str) -> str:
        return f"{self.base_uri.rstrip('/')}/objects/sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"

    def ensure_object(self, src_uri: str, sha256: str, data: bytes | None = None) -> str:
        """Land ``src_uri`` in its content-addressed slot, returning that slot.

        Idempotent: if the slot already holds the object (this run, a prior run,
        or another machine), nothing is transferred — content addressing makes
        the ``exists`` check a safe dedup, since identical content means
        identical key. Same-provider transfers are server-side copies;
        cross-provider transfers upload ``data`` (the bytes the caller already
        read to hash).
        """
        destination = self.object_uri(sha256)
        if not self.resolver.exists(destination):
            if self.server_side:
                self.resolver.server_copy(src_uri, destination)
            elif data is not None:
                self.resolver.write_bytes(destination, data)
            else:
                raise VisionPackError(
                    f"Cross-provider transfer of {src_uri!r} needs the object bytes to relay, "
                    "but none were provided."
                )
        return destination
