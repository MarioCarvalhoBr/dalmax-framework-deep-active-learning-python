# Use case: run the ablation study

Full specification: `experiments/ablation-study.md` — read it first, including its "Materialized
files" section mapping every run-table row to its actual params JSON and GPU script.

**Status update (2026-08-23, Phase 3 landed): every config and script is materialized, not just
JSON snippets in a spec file.** `files_config/ablations/*.json` (11 files, one per run-table row)
plus `files_config/ablations/micro/*.json` (CPU smoke mirrors) exist on disk, validated by
`tests/test_ablation_configs.py` and smoke-tested end-to-end by `make smoke-ablations` (11/11
passing on the local CPU-only dev notebook against `DATA/daninhas_micro/`).
`scripts/ablations/run_ablation_gpu_0.sh` / `run_ablation_gpu_1.sh` are the actual lab-machine
entry points (mirroring `run_pipe_gpu_0.sh`/`run_pipe_gpu_1.sh`'s style), splitting the 11 configs
across the two GPUs by expected cost. This use case is now purely operational: what remains is
running these two scripts on the lab machine (never locally — no GPU here, per
`.specs/infrastructure/execution-environments.md`) and aggregating with
`dalmax.reporting.ablation_report`.

## Order of operations

1. **Pre-flight: smoke-test locally first.** `make smoke-ablations` (or `bash
   scripts/ablations/smoke_ablations.sh`) runs all 11 configs against `DATA/daninhas_micro/` in
   under two minutes on CPU. Always green this before touching the lab machine — it exercises the
   exact same `RepresentationStrategy` code path (embedding provider, selection strategy, hierarchy
   depth) each full-scale config uses, just at micro scale.
2. **Metrics — already correct.** Every run through `dalmax.cli`/`demo.py` writes both weighted and
   macro F1 into `results.json` (`all_f1_score`/`all_f1_macro`) — see `research-rules/metrics.md`.
   The one remaining offline step: **pre-Phase-2 reference runs** (`results/dalmax{1,2}/`, an
   alternative source for §6.3's "RNHAL (full)" row if `stage_full.json`'s own re-run is not used)
   predate this fix and only have weighted F1 — recompute macro F1 for those specific runs offline
   from their `predictions.csv`, do not re-run them.
3. **Run the two lab scripts** (one per GPU, both survive an individual config's failure and log it
   instead of aborting the batch — check `results/ablations/gpu{0,1}_failures.log` afterward):
   ```bash
   # GPU 0: rep_full, rep_spatial, stage_no_representation, hier_L1, hier_L2b (5 configs)
   bash scripts/ablations/run_ablation_gpu_0.sh

   # GPU 1: rep_spectral, stage_full, stage_no_hierarchy, hier_L2a, hier_L3, hier_L4 (6 configs)
   bash scripts/ablations/run_ablation_gpu_1.sh
   ```
   Both sweep `SEEDS=(1 2 3)`, `n_query=100`, `n_round=8`, `n_init_labeled` left at the CLI default
   (100, per `experimental-protocol.md`), `--device cuda`, `--strategy_name RepresentationStrategy`.
   Each config writes to its own `results/ablations/<study>/<config>/` subtree (11 × 3 = 33 total
   runs), so `dalmax.reporting.ablation_report` can walk it directly — see
   `run_ablation_gpu_0.sh`'s header comment for the full GPU-split cost rationale (hierarchical
   selection with large/multi-level hierarchies dominates runtime, not training).
4. **Aggregate**:
   ```bash
   poetry run python -m dalmax.reporting.ablation_report \
     --root results/ablations --out paper_drafts/ablation_tables
   ```
   Writes `ablation_summary.csv` (final-round and across-rounds-mean macro/weighted F1, mean ± std
   across seeds, per config) and one booktabs `ablation_6_{1,2,3}.tex` / `.md` per sub-study — a
   config with zero discovered runs renders `TBD` rather than being silently dropped, so partial
   lab-machine progress is always visible in the table shape. `paper_drafts/` is gitignored, so
   re-running this after each lab-machine batch is safe and idempotent.

## Verification before trusting results

Before treating any ablation run's F1 as final, an `experiment-auditor`-style pass
(`.claude/agents/experiment-auditor.md`) should check: (a) `run_metadata.json` in the results
directory actually shows the intended `embedding`/`selection` config (not just the CLI args — the
full resolved config is right there now, no need to infer it from the params JSON filename), (b)
the cache file actually used (`results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{hash}.pkl`,
visible in the run log via `RepresentationStrategy`'s "Embedding matrix shape" log line) matches the
intended `Q`/`embedding_variant`/dataset combination, (c) the `hierarchy` used matches the intended
row of the §6.2 table exactly (`n_levels`, `n_clusters`, `sample_sizes` — also visible in
`run_metadata.json`'s `config.dataset.selection.hierarchy`), (d) `all_f1_macro` (not
`all_f1_score`, which stays weighted) is what gets reported — see `research-rules/metrics.md`, and
(e) `results/ablations/gpu{0,1}_failures.log` is empty (or every failure logged there has been
re-run and now succeeded) for the batch being reported on.

## Aggregation and paper hand-off

`dalmax.reporting.ablation_report` (step 4 above) is the current tool for this sweep specifically —
it supersedes `dalmax/reporting/chunk_results.py` /
`average_results.py` for the ablation family (those remain the tool for the main
`results/dalmax{1,2}/` sweeps, per `use-cases/generate-report.md`). Map the resulting
`ablation_6_{1,2,3}.tex` tables to the correct paper location per
`experiments/ablation-study.md`'s "Mapping to the paper's `\subsubsection`s" table.
