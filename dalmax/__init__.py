"""DalMax — Deep Active Learning Laboratory for UAV Weed Recognition (RNHAL).

This is the single top-level package (`.specs/architecture/refactor-plan.md`).
Introduced incrementally in Phase 2 alongside the legacy `core`/`utils`
packages it wrapped, `dalmax/` absorbed the remaining `core`/`utils` modules
in Phase 4 (moves + dead-code deletion) — there is no other Python package in
this repo. `trainer.py` (renamed from the historical `demo.py` on
2026-08-23) at the repo root is a thin shim that calls
`dalmax.cli.main()`.

This module must remain free of import-time side effects (no logging setup,
no file I/O, no training) — see `.claude/rules/code-quality.md`. The single
exception is `_ensure_matplotlib_backend()` below, which only touches one
environment variable and only when it is already broken.
"""

import importlib
import os

__version__ = "1.0.0"


def _ensure_matplotlib_backend() -> None:
    """Fall back to the headless ``Agg`` backend when ``MPLBACKEND`` is unusable.

    Notebook front-ends (Google Colab, Jupyter) export
    ``MPLBACKEND=module://matplotlib_inline.backend_inline`` to every child
    process. DalMax runs inside its own Poetry ``.venv``, where
    ``matplotlib_inline`` is not installed, so ``import matplotlib`` fails at
    import time with ``ValueError: Key backend: ... is not a valid value``
    (observed on Colab, 2026-08-25). Everything DalMax plots is written to
    files, so ``Agg`` is always correct here. A valid/absent ``MPLBACKEND`` is
    left untouched.
    """
    backend = os.environ.get("MPLBACKEND", "")
    prefix = "module://"
    if not backend.startswith(prefix):
        return
    try:
        importlib.import_module(backend[len(prefix) :])
    except ImportError:
        os.environ["MPLBACKEND"] = "Agg"


_ensure_matplotlib_backend()
