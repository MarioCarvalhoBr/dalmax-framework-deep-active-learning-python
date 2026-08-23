"""Thin dataset registry wrapping the legacy `utils.data` loaders and
`utils.dataset` handlers behind `get_dataset(config)`.

See `dalmax.data.registry` and `.specs/architecture/target-architecture.md`
§9 ("Registries replacing if/elif"). Importing this package must never have
side effects (no dataset loading, no file I/O).
"""

from __future__ import annotations
