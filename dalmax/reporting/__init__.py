"""Reporting utilities for DalMax experiments.

`dalmax.reporting.ablation_report` aggregates the Phase 3 ablation sweep's
`results/ablations/<study>/<config>/...` trees (see
`.specs/experiments/ablation-study.md`) into the paper's macro-F1 tables.
This package is new in Phase 3; it does not replace `utils/report/*.py`
(the pre-existing, unabstracted reporting scripts for the main
`results/dalmax{1,2}/` sweeps — see `.specs/architecture/refactor-plan.md`
Phase 4 for the eventual `utils/report/` -> `dalmax/reporting/` move).
"""

from __future__ import annotations
