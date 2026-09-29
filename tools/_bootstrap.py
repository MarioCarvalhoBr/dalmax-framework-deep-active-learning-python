"""Make the repo root importable when a script in `tools/` is run directly
(`python tools/trainer.py ...` puts `tools/`, not the repo root, on `sys.path`).

Import this module first (`import _bootstrap  # noqa: F401`); it only touches
`sys.path` and has no other side effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
