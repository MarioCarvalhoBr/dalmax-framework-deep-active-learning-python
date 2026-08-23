# Use case: run the ablation study

Full specification: `experiments/ablation-study.md` — read it first, including its exact
params-JSON snippets and CLI invocations for every row of all three sub-studies.

**Status update (2026-08-23, Phase 2 landed): all required code now exists.** Every capability this
file originally described as missing — embedding-variant slicing, configurable hierarchy without
the hardcoded `'DANINHAS'` key, the proportional-random flat k-means strategy, the ImageNet-ResNet
embedding provider — is implemented and tested (`dalmax/embeddings/`, `dalmax/selection/`,
`dalmax.query_strategies.RepresentationStrategy`; see
`.specs/architecture/refactor-plan.md`'s "ablation enablers checklist", fully checked off). This use
case is now the **actual** workflow, not a target one: every sub-study is runnable today via
config/CLI-only changes, on the lab machine (never locally — no GPU here, per
`.specs/infrastructure/execution-environments.md`). What is not yet done is the runs themselves.

## Order of operations

1. **Metrics fix first — done.** `research-rules/metrics.md`: macro-F1 is computed and persisted in
   `results.json` (`all_precision_macro`/`all_recall_macro`/`all_f1_macro`) for every run through
   `dalmax.cli`/`demo.py` from this commit onward. The one remaining action: **pre-Phase-2 reference
   runs** (`results/dalmax{1,2}/`, used by §6.3's "RNHAL (full)" row) predate this fix and only have
   weighted F1 in their `results.json` — recompute macro F1 for those specific runs offline from
   their `predictions.csv`, do not re-run them.
2. **§6.1 Representation ablation** — requires only the `embedding_variant` slicing capability
   (**done**: `dalmax/embeddings/variants.py`), no new strategy class, no new provider.
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_6_1_spatial.json --dataset_name DANINHAS \
     --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results results/ablation_representation/ --device cuda
   ```
   with the params JSON's `DANINHAS.embedding.variant` set to `spatial`, `spectral`, or `full` (one
   JSON file per variant — see `ablation-study.md` §6.1's exact config for the full JSON). 3
   variants × 3 seeds = 9 runs. Reuses the existing `Q=13` SSRAE cache automatically — slicing is a
   post-hoc operation on the cached full embedding (`dalmax/embeddings/cache.py`), so **no new
   feature-extraction pass happens** across the three variants for the same seed/pool.
3. **§6.2 Hierarchy ablation** — requires the config-driven hierarchy (**done**:
   `dalmax/selection/hierarchical_kmeans.py` takes `hierarchy` via constructor injection, no
   hardcoded dataset key) plus the `sample_sizes` derivation (**resolved**: see
   `ablation-study.md` §6.2's "sample_sizes semantics" note — `sample_sizes` only affects
   centroid-refinement resampling quality, `n_query` alone controls how many ids come out; exact
   values for all 5 rows are in that section's run table, TBD-flagged for advisor confirmation but
   safe to run as-is). 5 configs (`L=1,2a,2b,3,4`) × 3 seeds = 15 runs, `n_query=100` fixed, SSRAE
   full embedding.
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_6_2_L3.json --dataset_name DANINHAS \
     --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results results/ablation_hierarchy/ --device cuda
   ```
   one params JSON per row (`ablation-study.md` §6.2 has the exact `L=3` JSON to copy/adapt for the
   other four rows).
4. **§6.3 Stage-contribution ablation** — requires the ImageNet-ResNet embedding provider (**done**:
   `dalmax/embeddings/resnet_imagenet_provider.py::ResNetImageNetProvider`) and the
   proportional-random flat k-means strategy (**done**:
   `dalmax/selection/flat_kmeans_proportional.py::FlatKMeansProportionalRandom`, a genuinely new
   class, not a config-driven variant of `SSRAEKmeansSampling` — see ADR 0005). "RNHAL (full)" row
   needs **no new runs** — reuse `results/dalmax{1,2}/.../SSRAEKmeansHCSampling/` at `n_query=100`
   (recompute macro F1 offline, per step 1). The other two rows need new runs, all 3 seeds, at
   `n_query=100` (matching §6.1/§6.2 for consistency, since the source runs for "RNHAL (full)" are
   at that budget):
   ```bash
   # without representation module
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_6_3_resnet_hier.json --dataset_name DANINHAS \
     --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results results/ablation_stage_contribution/ --device cuda

   # without hierarchical module
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_6_3_ssrae_flat.json --dataset_name DANINHAS \
     --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results results/ablation_stage_contribution/ --device cuda
   ```
   See `ablation-study.md` §6.3's exact configs for the full JSON of both, and its one remaining
   TBD: whether `FlatKMeansProportionalRandom`'s cluster count (currently defaults to `n_query`,
   since `build_strategy` does not yet thread a separate `selection.n_clusters` params-JSON key to
   it) needs to be made independently configurable before this row is run — a small, well-scoped
   addition if the advisor confirms it is needed.

## Verification before trusting results

Before treating any ablation run's F1 as final, an `experiment-auditor`-style pass
(`.claude/agents/experiment-auditor.md`) should check: (a) `run_metadata.json` in the results
directory actually shows the intended `embedding`/`selection` config (not just the CLI args — the
full resolved config is right there now, no need to infer it from the params JSON filename), (b)
the cache file actually used (`results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{hash}.pkl`,
visible in the run log via `RepresentationStrategy`'s "Embedding matrix shape" log line) matches the
intended `Q`/`embedding_variant`/dataset combination, (c) the `hierarchy` used matches the intended
row of the §6.2 table exactly (`n_levels`, `n_clusters`, `sample_sizes` — also visible in
`run_metadata.json`'s `config.dataset.selection.hierarchy`), and (d) `all_f1_macro` (not
`all_f1_score`, which stays weighted) is what gets reported — see `research-rules/metrics.md`.

## Aggregation and paper hand-off

Once the runs for a given sub-study are complete, follow `use-cases/generate-report.md` to aggregate
across seeds, then map the resulting table to the correct paper location per
`experiments/ablation-study.md`'s "Mapping to the paper's `\subsubsection`s" table.
