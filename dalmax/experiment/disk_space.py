"""Disk-space policy of a campaign (2026-11-12).

On Colab the results tree lives on Google Drive, whose quota can fill up in the
middle of a GPU part (about 95 MB per run, mostly the checkpoint). Without a
gate the batch keeps training for hours and fails at every save. This module is
the single definition used by:

- `dalmax.campaign` (`run` refuses to start when the free space of the results
  file system is below `required_bytes(remaining runs)`, and stops the loop
  instead of continuing when a run fails for lack of space or the free space
  drops below one run);
- `dalmax.cli.main` (a trainer run that dies on a no-space error exits with
  `NO_SPACE_EXIT_CODE`, so the campaign can tell it from any other failure).

Importing this module has no side effects.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

# Bytes one run leaves on disk: measured 95.1 MB per leaf in the first campaign (222 leaves,
# 21.1 GB; the ResNet-50 checkpoint is 94.8 MB of it) plus the per-round history files of ADR 0016.
RUN_DISK_BYTES = 96 * 10**6
# Safety margin on top of the per-run size (embedding caches, logs, Drive accounting slack).
DISK_MARGIN = 1.2
# Exit code of `tools/trainer.py` when a run fails because the disk (or the Drive quota) is full.
NO_SPACE_EXIT_CODE = 28  # = errno.ENOSPC on Linux
_NO_SPACE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_NO_SPACE_MESSAGES = ("No space left on device", "Disk quota exceeded")


def required_bytes(n_runs: int, *, run_bytes: int = RUN_DISK_BYTES, margin: float = DISK_MARGIN) -> int:
    """Free bytes needed before starting `n_runs` runs: runs x per-run size x margin."""
    return int(n_runs * run_bytes * margin)


def existing_ancestor(path: str | Path) -> Path:
    """`path` itself or its nearest existing parent (a results root may not exist before the first run)."""
    current = Path(path).absolute()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def free_bytes(path: str | Path) -> int:
    """Free bytes of the file system holding `path` (follows symlinks, e.g. `results/` -> Drive)."""
    return shutil.disk_usage(existing_ancestor(path)).free


def file_system_id(path: str | Path) -> int:
    """Device id of the file system holding `path`: results roots on one device share their free space."""
    return existing_ancestor(path).stat().st_dev


def is_no_space_error(exc: BaseException) -> bool:
    """True for a full disk or quota anywhere in the exception chain (OSError errno, or the
    message torch/pickle wrap it in, e.g. `PytorchStreamWriter failed ... No space left on device`)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in _NO_SPACE_ERRNOS:
            return True
        if any(text in str(current) for text in _NO_SPACE_MESSAGES):
            return True
        current = current.__cause__ or current.__context__
    return False


def format_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"
