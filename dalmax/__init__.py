"""DalMax — Deep Active Learning Laboratory for UAV Weed Recognition (RNHAL).

This is the single top-level package (`.specs/architecture/refactor-plan.md`).
Introduced incrementally in Phase 2 alongside the legacy `core`/`utils`
packages it wrapped, `dalmax/` absorbed the remaining `core`/`utils` modules
in Phase 4 (moves + dead-code deletion) — there is no other Python package in
this repo. `trainer.py` (renamed from the historical `demo.py` on
2026-08-23) at the repo root is a thin shim that calls
`dalmax.cli.main()`.

This module must remain free of import-time side effects (no logging setup,
no file I/O, no training) — see `.claude/rules/code-quality.md`.
"""

__version__ = "1.0.0"
