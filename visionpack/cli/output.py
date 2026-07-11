from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

# The machine-readable contract every `--json` flag speaks. Bumped only on a
# breaking change to the envelope or an existing command's `data` shape —
# additive fields do not bump it. Consumers should check `schema` and the
# presence of `error` (plus the process exit code) rather than parse stderr.
SCHEMA_VERSION = 1


def emit_json(command: str, data: Any) -> None:
    """Print the success envelope for ``command`` to stdout.

    In JSON mode this must be the only thing on stdout: human-facing prints and
    progress bars are suppressed by the callers, so the output is always one
    parseable document.
    """
    _print({"schema": SCHEMA_VERSION, "command": command, "data": data})


def emit_json_error(command: str, error: Exception) -> None:
    """Print the error envelope for a command that raised.

    The process still exits non-zero; the envelope exists so a driving program
    gets a structured reason on stdout instead of scraping stderr.
    """
    _print(
        {
            "schema": SCHEMA_VERSION,
            "command": command,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    )


def _print(envelope: dict[str, Any]) -> None:
    print(json.dumps(envelope, indent=2, sort_keys=True, default=_default))


def _default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
