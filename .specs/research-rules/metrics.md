# Metric definitions

Exact definitions as implemented in `utils/dataset.py`/`utils/data.py`
(the `Data` class lives in `utils/data.py`; `utils/dataset.py` only contains
the `DANINHAS_Hander`/`CIFAR10_Handler` torch `Dataset` classes — there is
no `calc_metrics_sklearn` in `utils/dataset.py` itself, it is a method of
`Data` in `utils/data.py`; the task brief's pointer to
"`utils/dataset.py` (calc_metrics_sklearn)" is corrected here to the actual
location).

## `Data.cal_test_acc` (utils/data.py, custom accuracy)

```python
1.0 * (self.Y_test == preds).sum().item() / self.n_test
```

Plain elementwise-equality accuracy over the full test set (both tensors
cast to `int64` first). This is the value stored as `all_acc` in
`results.json` for every round — **not** `acc_skl` (see below).

## `Data.calc_metrics_sklearn` (utils/data.py:281-295)

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
  (`logger.warning(f'Final Accuracies (sklearn): {acc_skl}')` in `demo.py`)
  — it is not stored in `results.json`; only `Data.cal_test_acc`'s value
  populates the `all_acc` list that is persisted.

## Ablation study requires macro F1 — action required

`experiments/ablation-study.md` (per the advisor's request, and per
`prompt-master.md` §6) specifies **macro F1** as the ablation metric. Since
`calc_metrics_sklearn` uses `average='weighted'`, this is a real
discrepancy, not a naming nuance: macro F1 weights every class equally
regardless of support, so on this imbalanced test set (`BRACHIARIA`,
`COLONIAO` especially, see `dataset-protocol.md`) macro and weighted F1 can
diverge materially. Two implementation paths, both viable, described in
`ablation-study.md`:

1. Add a macro-F1 computation (`f1_score(..., average='macro')`) alongside
   the existing weighted one in `Data.calc_metrics_sklearn`, and persist
   both in `results.json` going forward (non-breaking addition).
2. Recompute macro F1 offline from `predictions.csv` (`Real Class`,
   `Predicted Class` columns, already saved by `demo.py` for every run) —
   requires no new training runs for the existing RNHAL reference numbers
   used in ablation §6.3.

**Do not report the existing `all_f1_score` values from `results.json` as
"macro F1" in the paper** — they are weighted F1. Any table mixing the two
without labeling them explicitly would misrepresent the ablation.

## `Data.calc_metrics_manual` (utils/data.py:259-279, unused fallback)

A hand-rolled binary-style TP/TN/FP/FN precision/recall/F1 computation using
tensor bitwise ops (`&`, `~`). This only makes sense for boolean/binary
labels and is **not called anywhere in `demo.py`** — appears to be dead code
retained from an earlier binary-classification iteration of the project.
Not used for any reported metric; flagged here so it is not mistaken for
the active metric path.

## Rounding / aggregation across seeds

`utils/report/2_report_build_chunk_results.py`,
`3_cm_build_average.py`, and `4_report_build_average_results.py` all define
the same `MetricsType` enum (`ALL_ACC`, `ALL_PRECISION`, `ALL_RECALL`,
`ALL_F1_SCORE`) and aggregate per `(n_query config, method)` across the
`SEED_*` folders (mean across seeds per round, based on the header/args
inspected: `--input_dir results/dalmax1/daninhas_full/[...] --pattern
SEED*`). **Exact rounding precision and whether it's mean-only or
mean±std is TBD** — only the first ~40 lines of each script were read in
this batch (headers + `create_csv_tables` signature); the full aggregation
body was not inspected. Confirm before quoting a specific mean±std format
in the paper.
