---
name: results-reporting
description: How to use the dalmax/reporting/ scripts to go from raw results directories to averaged metrics and LaTeX-ready tables.
---

# Results reporting

`dalmax/reporting/` contains the pipeline that turns raw per-run `results.json`
files (written by `trainer.py` under
`results/<run>/<dataset>/SEED_<seed>/NQ_<n_query>_NIL_<n_init>_NR_<n_round>_NE_<n_epoch>/<strategy>/`)
into aggregated tables and plots. Run the scripts in the order below; each
script's own header comment (`# Example usage: ...`) is the authoritative
source for its exact CLI flags — read it before running. Every script is
runnable either directly (`poetry run python dalmax/reporting/<name>.py`) or as a module
(`poetry run python -m dalmax.reporting.<name>`).

(Moved and renamed from `utils/report/{1_cm_extract_from_pdf,
2_report_build_chunk_results, 3_cm_build_average,
4_report_build_average_results}.py` in refactor Phase 4 — the numbered
prefixes are gone, since a name starting with a digit can't be used with
`python -m`.)

## Pipeline, in order

1. **`dalmax/reporting/extract_confusion_matrices.py`** (was
   `1_cm_extract_from_pdf.py`) — extracts the confusion matrix values from a
   run's `confusion_matrix.pdf` (via `PyPDF2`, regex between "True" and
   "Confusion Matrix" in the extracted text) when the raw numbers are only
   available as a rendered PDF and not already in `results.json`. Only needed
   as a fallback for older/PDF-only results; skip it when `results.json`
   already has what you need.

2. **`dalmax/reporting/chunk_results.py`** (was
   `2_report_build_chunk_results.py`)
   (`poetry run python -m dalmax.reporting.chunk_results --input_dir results/dalmax1/daninhas_full/ --pattern SEED*`) —
   walks a results tree matching `--pattern` (e.g. `SEED*`), reads each run's
   `results.json`, and builds per-seed CSV tables and plots for the four
   metrics it tracks (`MetricsType`: `all_acc`, `all_precision`, `all_recall`,
   `all_f1_score`), organized by `NQ_*` (n_query) configuration.

3. **`dalmax/reporting/average_confusion_matrices.py`** (was
   `3_cm_build_average.py`)
   (`poetry run python -m dalmax.reporting.average_confusion_matrices --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`) —
   averages confusion matrices across seeds for each strategy/`n_query`
   configuration.

4. **`dalmax/reporting/average_results.py`** (was
   `4_report_build_average_results.py`)
   (`poetry run python -m dalmax.reporting.average_results --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`) —
   builds the final across-seed averaged metrics tables/plots (same
   `MetricsType` set as step 2, now averaged over `SEEDS`), the form most
   directly useful for a paper table.

5. **`dalmax/reporting/build_method_metrics.py`** — a narrower, single-method
   utility: computes the average of a metric for one strategy at one round and
   one `n_query`, across seeds matching a glob pattern. Example usage (from its
   own docstring):
   ```bash
   poetry run python dalmax/reporting/build_method_metrics.py --method MarginSampling --round 8 --nq 100
   poetry run python dalmax/reporting/build_method_metrics.py --method LeastConfidence --round 5 --nq 50 --input_folder ./resultados
   poetry run python dalmax/reporting/build_method_metrics.py --method BALDDropout --round 3 --nq 10 --seed "EXPERIMENTO_*"
   ```
   Use this for a quick one-off number instead of the full pipeline.

6. **`dalmax/reporting/plot_results_dir.py`**
   (`poetry run python dalmax/reporting/plot_results_dir.py --dir_input results/<some_run_dir>`) —
   lists all run subfolders under `--dir_input`, reads each `results.json`, and
   produces summary plots directly from a flat results directory (does not
   require the `SEED_*` / `NQ_*` nesting the numbered pipeline expects — TBD
   confirm exact directory shape it assumes; read the script if unsure before
   pointing it at a non-standard layout).

## Notes

- All of steps 2-4 depend on `pandas`, `numpy`, `matplotlib`, `seaborn` —
  `poetry install` installs all of them (`pandas` was historically missing
  from the project's dependency list; see `.specs/quality/known-issues.md`).
- Never point these scripts' output at a path that would overwrite an existing
  raw `results.json` or plot from a prior run — `.claude/rules/data-safety.md`
  treats `results/` as append-only. Check each script's `argparse` defaults for
  where it writes derived output before running it against a shared results
  tree.
- There is also `dalmax/reporting/ablation_report.py`, a separate script that
  aggregates `results/ablations/` for the Phase 3 ablation studies — see
  `.specs/experiments/ablation-study.md`, not part of this numbered pipeline.
- `.claude/commands/results-report.md` wraps steps 2-4 (and optionally 5) as a
  single command for a given results directory.
