"""VisionPack public Python API.

``Project``/``Dataset`` are the low-level handles; the supported programmatic
surface is the SDK (:mod:`visionpack.sdk`), which adds locking, stable return
shapes, and snapshot views:

    from visionpack.sdk import VisionPackClient
"""

from visionpack.core.project import Dataset, Project

__all__ = ["Dataset", "Project"]
