"""Thin model registry wrapping `dalmax.models.base.DeepLearning` and the
per-dataset model classes behind `get_network(config, device)`.

See `dalmax.models.registry`. Importing this package must never have side
effects (no model construction, no weight loading).
"""

from __future__ import annotations
