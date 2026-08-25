"""`import dalmax` must survive a notebook-style, unusable ``MPLBACKEND``.

Reproduces the Colab failure of 2026-08-25: the notebook front-end exports
``MPLBACKEND=module://matplotlib_inline.backend_inline`` to subprocesses, but
that module is not installed in the Poetry ``.venv``, so ``import matplotlib``
raised at import time. ``dalmax.__init__._ensure_matplotlib_backend`` falls
back to ``Agg`` in exactly that case and leaves a valid value alone.
"""

from __future__ import annotations

import os
import subprocess
import sys

_PROBE = (
    "import dalmax, os, matplotlib, matplotlib.pyplot as plt; "
    "print(os.environ.get('MPLBACKEND'), matplotlib.get_backend())"
)


def _run(env_backend: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    if env_backend is not None:
        env["MPLBACKEND"] = env_backend
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_unimportable_module_backend_falls_back_to_agg() -> None:
    line = _run("module://matplotlib_inline.backend_inline")
    assert line.split()[0] == "Agg"
    assert "agg" in line.split()[1].lower()


def test_valid_explicit_backend_is_left_untouched() -> None:
    assert _run("Agg").split()[0] == "Agg"


def test_absent_backend_is_left_absent() -> None:
    assert _run(None).split()[0] == "None"
