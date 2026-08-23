"""Shared pytest configuration and fixtures for the DalMax test suite.

Notes
-----
- The repository does not install ``dalmax`` as a package (it is a plain
  top-level package used in-place, see ``pyproject.toml``'s
  ``package-mode = false``), so the repo root must be on ``sys.path`` for
  ``import dalmax...`` to work regardless of how pytest is invoked
  (``poetry run pytest``, ``pytest`` from a subdirectory, etc).
- Markers (``gpu``, ``dataset``, ``slow``) are already declared in
  ``pyproject.toml`` under ``[tool.pytest.ini_options]``. The registration
  below is defensive: if a future session runs this suite against a
  ``pyproject.toml``/``pytest.ini`` that does not declare them (e.g. a
  stripped-down CI config), we still register them here instead of letting
  pytest fail with "unknown marker" warnings/errors. Registering twice is
  harmless, but we guard against exact duplicates anyway.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MARKERS = {
    "gpu": "gpu: marks tests that require a CUDA GPU",
    "dataset": "dataset: marks tests that require the DATA/ dataset to be present",
    "slow": "slow: marks slow-running tests (full pipeline / training)",
}


def pytest_configure(config):
    existing = {line.split(":", 1)[0].strip() for line in config.getini("markers")}
    for name, definition in _MARKERS.items():
        if name not in existing:
            config.addinivalue_line("markers", definition)


@pytest.fixture
def tiny_rgb_image():
    """A tiny, deterministic 16x16x3 uint8 RGB image for fast CPU-only tests."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(low=0, high=256, size=(16, 16, 3), dtype=np.uint8)
