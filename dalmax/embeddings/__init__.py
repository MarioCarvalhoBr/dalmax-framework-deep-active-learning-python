"""Embedding provider abstraction (Phase 2 of the refactor).

Wraps the vendored feature extractors (`dalmax.tools.SSRAE`, `dalmax.tools.VCTex`)
and a new ImageNet-pretrained ResNet50 provider behind a common
`EmbeddingProvider` interface, plus an explicitly-keyed on-disk cache. See
`.specs/architecture/target-architecture.md` §2/§4 and
`.claude/rules/reproducibility.md` ("Embedding cache discipline").

Importing this package must never have side effects (no file I/O, no model
downloads, no GPU calls) — provider construction (which may download/load
model weights) only happens when a provider class is instantiated by caller
code, never at import time.
"""

from __future__ import annotations
