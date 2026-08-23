---
name: results-reporting
description: How to use the utils/report/ scripts to go from raw results directories to averaged metrics and LaTeX-ready tables.
---

# Results reporting

`utils/report/` contains the pipeline that turns raw per-run `results.json`
files (written by `demo.py` under
`results/<run>/<dataset>/SEED_<seed>/NQ_<n_query>_NIL_<n_init>_NR_<n_round>_NE_<n_epoch>/<strategy>/`)
into aggregated tables and plots. Run the scripts in the numbered order below;
each script's own header comment (`# Example usage: ...`) is the authoritative
source for its exact CLI flags — read it before running.

## Pipeline, in order

1. **`utils/report/1_cm_extract_from_pdf.py`** — extracts the confusion matrix
   values from a run's `confusion_matrix.pdf` (via `PyPDF2`, regex between
   "True" and "Confusion Matrix" in the extracted text) when the raw numbers
   are only available as a rendered PDF and not already in `results.json`.
   Only needed as a fallback for older/PDF-only results; skip it when
   `results.json` already has what you need.

2. **`utils/report/2_report_build_chunk_results.py`**
   (`python 2_report_build_chunk_results.py --input_dir results/dalmax1/daninhas_full/ --pattern SEED*`) —
   walks a results tree matching `--pattern` (e.g. `SEED*`), reads each run's
   `results.json`, and builds per-seed CSV tables and plots for the four
   metrics it tracks (`MetricsType`: `all_acc`, `all_precision`, `all_recall`,
   `all_f1_score`), organized by `NQ_*` (n_query) configuration.

3. **`utils/report/3_cm_build_average.py`**
   (`python3 3_cm_build_average.py --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`) —
   averages confusion matrices across seeds for each strategy/`n_query`
   configuration.

4. **`utils/report/4_report_build_average_results.py`**
   (`python 4_report_build_average_results.py --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`) —
   builds the final across-seed averaged metrics tables/plots (same
   `MetricsType` set as step 2, now averaged over `SEEDS`), the form most
   directly useful for a paper table.

5. **`utils/report/build_method_metrics.py`** — a narrower, single-method
   utility: computes the average of a metric for one strategy at one round and
   one `n_query`, across seeds matching a glob pattern. Example usage (from its
   own docstring):
   ```bash
   python3 utils/report/build_method_metrics.py --method MarginSampling --round 8 --nq 100
   python3 utils/report/build_method_metrics.py --method LeastConfidence --round 5 --nq 50 --input_folder ./resultados
   python3 utils/report/build_method_metrics.py --method BALDDropout --round 3 --nq 10 --seed "EXPERIMENTO_*"
   ```
   Use this for a quick one-off number instead of the full pipeline.

6. **`utils/report/plot_results_dir.py`**
   (`python utils/report/plot_results_dir.py --dir_input results/<some_run_dir>`) —
   lists all run subfolders under `--dir_input`, reads each `results.json`, and
   produces summary plots directly from a flat results directory (does not
   require the `SEED_*` / `NQ_*` nesting the numbered pipeline expects — TBD
   confirm exact directory shape it assumes; read the script if unsure before
   pointing it at a non-standard layout).

## Notes

- All of steps 2-4 depend on `pandas`, `numpy`, `matplotlib`, `seaborn` —
  confirm these are installed (`pandas` was historically missing from
  `requirements.txt`; see `.specs/quality/known-issues.md`).
- Never point these scripts' output at a path that would overwrite an existing
  raw `results.json` or plot from a prior run — `.claude/rules/data-safety.md`
  treats `results/` as append-only. Check each script's `argparse` defaults for
  where it writes derived output before running it against a shared results
  tree.
- `.claude/commands/results-report.md` wraps steps 2-4 (and optionally 5) as a
  single command for a given results directory.
