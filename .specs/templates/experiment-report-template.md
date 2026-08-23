# Experiment Report — <short run name>

Status: Draft | Final — date YYYY-MM-DD

## Identification

| Field | Value |
|---|---|
| Dataset | `daninhas_full` \| `CIFAR10` \| ... |
| Strategy | e.g. `SSRAEKmeansHCSampling` |
| Seed(s) | e.g. `1, 2, 3` |
| `n_init_labeled` | |
| `n_query` | |
| `n_round` | |
| `n_epoch` | |
| params JSON file | e.g. `params_df_gpu_0.json` |
| Git commit | `<hash>` (from `run_metadata.json` once Phase 2 lands; `TBD` for pre-refactor runs) |
| Results directory | `results/<...>/SEED_<seed>/NQ_<n_query>_NIL_<n_init>_NR_<n_round>_NE_<n_epoch>/<strategy>/` |
| Execution environment | local dev \| lab machine (GPU 0/1) \| Colab |

## Configuration snapshot

Paste (or link to) the exact `config_kmh` / `optimizer_args` / `train_args` / `test_args` block used,
and any `embedding_variant` or ablation-specific setting (`.specs/experiments/ablation-study.md`).
If `run_metadata.json` exists for this run, prefer quoting it verbatim over retyping values.

```json
<paste relevant params block>
```

## Metrics table

Per-round metrics (from `results.json`), one row per round:

| Round | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|---|---|---|---|---|---|
| 0 | | | | | |
| 1 | | | | | |
| ... | | | | | |

Note: `calc_metrics_sklearn` currently reports `average='weighted'` (see
`.specs/quality/known-issues.md` KI-14); fill the macro-F1 column only once that fix has landed or
been computed manually from `predictions.csv`.

## Aggregation across seeds (if reporting a multi-seed summary)

| Strategy | Mean F1 (macro) | Std F1 (macro) | Seeds |
|---|---|---|---|
| | | | |

## Artifacts

- [ ] `results.json`
- [ ] `predictions.csv`
- [ ] `confusion_matrix.pdf`
- [ ] `accuracy.pdf` / `precision.pdf` / `recall.pdf` / `f1_score.pdf`
- [ ] `saved_model.pth`
- [ ] `log-dalmax.log`
- [ ] `run_metadata.json` (post-Phase-2 runs only)

## Observations

Free-text: anything notable about convergence, class imbalance effects, cache hits/misses,
anomalies worth flagging to the advisor. `TBD` if this report is generated before the run finished.

## Related

- `.specs/experiments/experimental-protocol.md`
- `.specs/experiments/ablation-study.md` (if this run is part of an ablation)
- `.specs/experiments/baseline-results.md` (if this run is a baseline reference point)
