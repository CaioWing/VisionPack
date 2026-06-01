class VisionPackError(Exception):
    """Base exception for actionable VisionPack errors."""


class ProjectNotFoundError(VisionPackError):
    """Raised when a command is executed outside a VisionPack project."""


class ManifestError(VisionPackError):
    """Raised when visionpack.yaml cannot be read or validated."""


class FormatError(VisionPackError):
    """Raised when an input dataset format cannot be parsed."""
