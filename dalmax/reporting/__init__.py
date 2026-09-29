"""Reporting utilities for DalMax experiments.

`dalmax.reporting.campaign_report` builds the per-paper tables and mean confusion matrices of
the campaign (`.specs/experiments/campaign.md`); `dalmax.reporting.leaf_check` is the artifact-completeness
check of one run directory.

The rest of this package (`extract_confusion_matrices.py`, `chunk_results.py`,
`average_confusion_matrices.py`, `average_results.py`, `build_method_metrics.py`,
`plot_results_dir.py`) is the pre-existing, unabstracted reporting pipeline
for the main `results/dalmax{1,2}/` sweeps, moved here unchanged (aside from
import-line/rename edits) from `utils/report/` in refactor Phase 4
(`.specs/architecture/refactor-plan.md`) — see
`.claude/skills/results-reporting/SKILL.md` for the old-name -> new-name
mapping and usage.
"""

from __future__ import annotations
