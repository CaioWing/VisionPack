from __future__ import annotations

import threading
from dataclasses import dataclass, field

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

    Membership is resolved with one **prefix listing** of the target CAS on
    first use (rclone's fast-list approach), not a per-object existence check:
    at 100k objects that is ~100 paginated LISTs instead of 100k HEADs. The set
    is safe to consult from ingest worker threads.
    """

    base_uri: str
    resolver: Resolver
    server_side: bool = True
    _present: set[str] | None = field(default=None, init=False)
    _mutex: threading.Lock = field(default_factory=threading.Lock, init=False)

    def object_uri(self, sha256: str) -> str:
        return f"{self.base_uri.rstrip('/')}/objects/sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"

    def ensure_object(self, src_uri: str, sha256: str, data: bytes | None = None) -> str:
        """Land ``src_uri`` in its content-addressed slot, returning that slot.

        Idempotent: if the slot already holds the object (this run, a prior run,
        or another machine), nothing is transferred — content addressing makes
        the membership check a safe dedup, since identical content means
        identical key. Same-provider transfers are server-side copies;
        cross-provider transfers upload ``data`` (the bytes the caller already
        read to hash) and verify the landed size.
        """
        destination = self.object_uri(sha256)
        if sha256 in self._known():
            return destination
        if self.server_side:
            self.resolver.server_copy(src_uri, destination)
        elif data is not None:
            self.resolver.write_bytes(destination, data)
            self._verify_upload(destination, len(data))
        else:
            raise VisionPackError(
                f"Cross-provider transfer of {src_uri!r} needs the object bytes to relay, "
                "but none were provided."
            )
        with self._mutex:
            assert self._present is not None  # _known() above populated it
            self._present.add(sha256)
        return destination

    def _known(self) -> set[str]:
        """The sha256s already present in the target, listed once per sync run.

        Two workers racing on the *same new* hash may both transfer it — the
        write is idempotent (same bytes, same key), so that costs one duplicate
        upload at worst and never corrupts the CAS.
        """
        with self._mutex:
            if self._present is None:
                prefix = f"{self.base_uri.rstrip('/')}/objects/sha256"
                try:
                    refs = self.resolver.list_files(prefix)
                except VisionPackError:
                    refs = []  # target CAS not created yet: everything is new
                # Object keys are bare sha256s (no extension), so stem == hash.
                self._present = {ref.stem for ref in refs}
            return self._present

    def _verify_upload(self, destination: str, expected_size: int) -> None:
        # A relayed upload transits the client, so confirm the provider landed
        # every byte before the index starts pointing at the slot. (Server-side
        # copies don't need this: the provider guarantees copy integrity.)
        landed = self.resolver.stat(destination)
        if landed.size != expected_size:
            raise VisionPackError(
                f"Relay upload to {destination!r} landed {landed.size} bytes, expected {expected_size}; "
                "the object was not indexed — re-run the sync."
            )
