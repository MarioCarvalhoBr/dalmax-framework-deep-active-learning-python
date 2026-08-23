"""DalMax — Deep Active Learning Laboratory for UAV Weed Recognition (RNHAL).

This is the new top-level package introduced in the Phase 2 refactor
(`.specs/architecture/refactor-plan.md`). Legacy code under `core/` and
`utils/` keeps working in place; new modules are added here incrementally
and `demo.py` is routed through `dalmax/` in a later integration step.

This module must remain free of import-time side effects (no logging setup,
no file I/O, no training) — see `.claude/rules/code-quality.md`.
"""

__version__ = "1.0.0"
