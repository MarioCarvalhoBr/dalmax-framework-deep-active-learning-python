# Metric definitions

Exact definitions as implemented in `dalmax/data/handlers.py`/`dalmax/data/datasets.py`
(was `utils/dataset.py`/`utils/data.py` before Phase 4's move; both files were deleted,
their logic moved, not rewritten). The `Data` class lives in `dalmax/data/datasets.py`;
`dalmax/data/handlers.py` only contains
the `DANINHAS_Hander`/`CIFAR10_Handler` torch `Dataset` classes — there is
no `calc_metrics_sklearn` in that file, it is a method of
`Data` in `dalmax/data/datasets.py`; the task brief's original pointer to
"`utils/dataset.py` (calc_metrics_sklearn)" was corrected here to the actual
location even before Phase 4's move.

## `Data.cal_test_acc` (dalmax/data/datasets.py, custom accuracy)

```python
1.0 * (self.Y_test == preds).sum().item() / self.n_test
```

Plain elementwise-equality accuracy over the full test set (both tensors
cast to `int64` first). This is the value stored as `all_acc` in
`results.json` for every round — **not** `acc_skl` (see below).

## `Data.calc_metrics_sklearn` (dalmax/data/datasets.py, was `utils/data.py:281-295` before Phase 4) — legacy, unused by `dalmax`

**Phase 2 note**: this method is unchanged and still present, but
`dalmax/experiment/runner.py` calls the new `Data.calc_metrics` (below)
instead, not this one. Kept for backward compatibility with any direct
caller; described here for the historical record of what pre-Phase-2
`results.json` files were computed with.

```python
accuracy  = accuracy_score(self.Y_test, preds)
precision = precision_score(self.Y_test, preds, average='weighted', zero_division=0)
recall    = recall_score(self.Y_test, preds, average='weighted', zero_division=0)
f1        = f1_score(self.Y_test, preds, average='weighted', zero_division=0)
```

- **`average='weighted'` for precision, recall, and F1** — this is the
  actual, verified averaging scheme used everywhere in the codebase today,
  for every existing result under `results/`. It is **not macro-averaged**.
  Weighted averaging computes the per-class metric and averages it weighted
  by each class's support in `Y_test`, which — given the class imbalance in
  `daninhas_full` (see `dataset-protocol.md`: test-set counts range from
  155 to 970 images per class) — will systematically favor performance on
  the majority classes (`GRAMINEA`, `MAMONA`) over minority ones
  (`OUTRAS_FOLHAS_LARGAS`, `COLONIAO`).
- `zero_division=0` — classes with no predicted samples contribute 0 rather
  than raising/warning.
- The returned `accuracy` (`acc_skl`, sklearn's `accuracy_score`) is
  computed every round but **only the final round's value is logged**
  (`logger.warning(f'Final Accuracies (sklearn): {acc_skl}')` — this was a
  `demo.py`-only log line, tied to the legacy `calc_metrics_sklearn` call path;
  since `dalmax` never calls `calc_metrics_sklearn` and `demo.py` is now a
  12-line shim, this specific log line no longer runs anywhere — confirmed
  absent from `dalmax/` by grep)
  — it is not stored in `results.json`; only `Data.cal_test_acc`'s value
  populates the `all_acc` list that is persisted.

## Macro F1 is now computed (resolved 2026-08-23, Phase 2) — which metric to report

`experiments/ablation-study.md` (per the advisor's request, and per
`prompt-master.md` §6) specifies **macro F1** as the ablation metric. Since
`calc_metrics_sklearn` uses `average='weighted'`, this was a real
discrepancy, not a naming nuance: macro F1 weights every class equally
regardless of support, so on this imbalanced test set (`BRACHIARIA`,
`COLONIAO` especially, see `dataset-protocol.md`) macro and weighted F1 can
diverge materially. Implementation path (1) below was chosen and landed:

1. **Done.** `dalmax/data/datasets.py::Data.calc_metrics` (new method, added alongside
   the untouched `calc_metrics_sklearn`) computes accuracy plus weighted
   *and* macro-averaged precision/recall/F1 in one call:
   ```python
   {
       "acc": accuracy_score(...),
       "precision_weighted": ..., "recall_weighted": ..., "f1_weighted": ...,
       "precision_macro": ..., "recall_macro": ..., "f1_macro": ...,
   }
   ```
   `dalmax/experiment/runner.py::ExperimentRunner._record_round` calls this
   for every round; `dalmax/experiment/reporter.py::_write_results_json`
   persists the macro values as `all_precision_macro`/`all_recall_macro`/
   `all_f1_macro` in `results.json`, additive to the unchanged legacy keys
   (`all_precision`/`all_recall`/`all_f1_score`, still weighted). Every run
   through `dalmax.cli`/`demo.py` from this commit onward has both.
2. **Still needed for pre-Phase-2 `results.json` files** (e.g.
   `results/dalmax1/`, `results/dalmax2/`, used as-is by ablation §6.3's
   "RNHAL (full)" row): recompute macro F1 offline from `predictions.csv`
   (`Real Class`, `Predicted Class` columns, already saved by `demo.py` for
   every run) — no re-training needed, since those runs predate the
   `calc_metrics` addition and only have weighted F1 in their `results.json`.

**Reporting rule going forward**: for the ablation study and any paper table
claiming "macro F1", use `results.json`'s `all_f1_macro` (Phase 2+ runs) or
an offline `predictions.csv` recomputation (pre-Phase-2 runs) —
**never `all_f1_score`**, which remains weighted F1 for both old and new
runs (unchanged key, unchanged meaning). Any table mixing weighted and macro
values without labeling them explicitly would misrepresent the ablation.

## `Data.calc_metrics_manual` (dalmax/data/datasets.py, was `utils/data.py:259-279` before Phase 4, unused fallback)

A hand-rolled binary-style TP/TN/FP/FN precision/recall/F1 computation using
tensor bitwise ops (`&`, `~`). This only makes sense for boolean/binary
labels and is **not called anywhere in `demo.py` or `dalmax/`** — appears to be dead code
retained from an earlier binary-classification iteration of the project.
Not used for any reported metric; flagged here so it is not mistaken for
the active metric path.

## Rounding / aggregation across seeds

`dalmax/reporting/chunk_results.py`,
`average_confusion_matrices.py`, and `average_results.py` (renumbered/renamed from
`utils/report/2_report_build_chunk_results.py`/`3_cm_build_average.py`/
`4_report_build_average_results.py` in Phase 4) all define
the same `MetricsType` enum (`ALL_ACC`, `ALL_PRECISION`, `ALL_RECALL`,
`ALL_F1_SCORE`) and aggregate per `(n_query config, method)` across the
`SEED_*` folders (mean across seeds per round, based on the header/args
inspected: `--input_dir results/dalmax1/daninhas_full/[...] --pattern
SEED*`). **Exact rounding precision and whether it's mean-only or
mean±std is TBD** — only the first ~40 lines of each script were read in
this batch (headers + `create_csv_tables` signature); the full aggregation
body was not inspected. Confirm before quoting a specific mean±std format
in the paper.
