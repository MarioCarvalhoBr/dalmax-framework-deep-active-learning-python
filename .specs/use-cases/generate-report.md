# Use case: generate a report from a results directory

The `utils/report/` pipeline turns raw per-run `results.json`/
`predictions.csv` output into seed-averaged tables and plots. Scripts are
numbered `1_` through `4_` suggesting an intended order, plus two
standalone helpers (`build_method_metrics.py`, `plot_results_dir.py`).
Only script **headers/signatures** (first ~40 lines each) were read for
this batch — bodies beyond that were not fully traced; treat the ordering
and I/O below as verified only to that depth, and confirm exact CSV/plot
schemas before depending on them for the paper.

## Pipeline order (as suggested by filenames and doc-comments)

1. **`1_cm_extract_from_pdf.py`** — extracts the confusion-matrix numbers
   back out of the `confusion_matrix.pdf` a run produced, by regex-matching
   the text between `"True"` and `"Confusion Matrix"` in the PDF's
   extracted text (via `PyPDF2`). Useful when only the PDF survived and the
   underlying array wasn't otherwise persisted. Not needed for runs where
   the raw prediction data (`predictions.csv`) is available, since a
   confusion matrix can be recomputed directly from it.
2. **`2_report_build_chunk_results.py`** — example usage in its header:
   `python 2_report_build_chunk_results.py --input_dir results/dalmax1/daninhas_full/ --pattern SEED*`.
   Reads per-seed `results.json` files, groups by `n_query` config
   (`NQ_*` folder) and by method (strategy), and builds CSV tables per the
   `MetricsType` enum (`Accuracy`, `Precision`, `Recall`, `F1-score` mapped
   to `all_acc`/`all_precision`/`all_recall`/`all_f1_score`). This is the
   likely source of `results/dalmax1/daninhas_full/data_results.json` (not
   confirmed by reading the full body — TBD).
3. **`3_cm_build_average.py`** — example usage:
   `python3 3_cm_build_average.py --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`.
   Same `MetricsType`/`create_csv_tables` structure as script 2; averages
   confusion-matrix-derived data across seeds. Note its default `--input_dir`
   points at `.../results/` (the report-pipeline's own output subfolder),
   i.e. it consumes script 2's output, not the raw `SEED_*` run
   directories directly.
4. **`4_report_build_average_results.py`** — example usage:
   `python 4_report_build_average_results.py --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*`.
   Same structure again; produces the final seed-averaged results — this is
   almost certainly what populates
   `results/dalmax1/daninhas_full/results/AVERAGES/NQ_{10,50,100}_.../`.

**TBD**: the exact division of labor between scripts 2, 3, and 4 (why three
near-identical `MetricsType`/`create_csv_tables` definitions exist) was not
fully resolved — their bodies beyond the first ~40 lines were not read in
this batch. Before relying on this pipeline for the ablation study's
aggregation (`use-cases/run-ablation.md`), read all three scripts in full
and confirm which one is authoritative for the final seed-averaged macro-F1
table, or consider consolidating them (candidate refactor-plan item, see
`future/ideas.md`).

## Standalone helpers

- **`build_method_metrics.py`** — computes the across-seed average for one
  `(method, round, n_query)` triple directly, without going through the
  `1_`–`4_` pipeline. Example from its own docstring:
  ```
  python3 main.py --method MarginSampling --round 8 --nq 100
  python3 main.py --method LeastConfidence --round 5 --nq 50 --input_folder ./resultados
  python3 main.py --method BALDDropout --round 3 --nq 10 --seed "EXPERIMENTO_*"
  ```
  (Note: its own examples say `python3 main.py`, not
  `build_method_metrics.py` — TBD whether this file is meant to be run
  directly or was copy-pasted from a differently-named script; verify the
  actual entry point before use.) Builds the `NQ_{nq}_NIL_100_NR_{round}_NE_10`
  pattern internally, hardcoding `NIL_100` and `NE_10` — matches the
  observed protocol default (`n_init_labeled=100`, `n_epoch=10`) but will
  silently mismatch if either changes.
- **`plot_results_dir.py`** — example usage:
  `python utils/plot_results_dir.py --dir_input results/new_dalmax_balanceado_train_10_epochs_10_n_query`
  (a directory name pattern not otherwise seen in this batch's observed
  `results/` listing — likely from an earlier project iteration). Reads
  every subfolder's `results.json` directly (one level, not `SEED_*`
  nested), extracts `all_acc` per strategy, and plots — simplest of the
  report tools, useful for a quick single-seed sanity plot rather than a
  seed-averaged report.

## Recommended invocation for a finished sweep

```
python utils/report/2_report_build_chunk_results.py --input_dir results/<run>/daninhas_full/ --pattern "SEED*"
python utils/report/3_cm_build_average.py           --input_dir results/<run>/daninhas_full/results/ --pattern "SEED*"
python utils/report/4_report_build_average_results.py --input_dir results/<run>/daninhas_full/results/ --pattern "SEED*"
```

then inspect `results/<run>/daninhas_full/results/AVERAGES/` for the final
per-`n_query` seed-averaged tables/plots. This can run on the **local
machine** (CPU-only, no dataset needed — only reads already-produced
`results/` JSON/CSV files) once results are copied back from the lab
machine/Colab, per `infrastructure/execution-environments.md`.
