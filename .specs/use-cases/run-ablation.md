# Use case: run the ablation study

Full specification: `experiments/ablation-study.md` — read it first. This
file is the operational sequencing: what to implement, in what order, and
how to invoke it once implemented. **None of the required code exists yet**
(embedding-variant slicing, configurable hierarchy without the hardcoded
`'DANINHAS'` key, the proportional-random flat k-means strategy, the
ImageNet-ResNet embedding provider) — this use case describes the target
workflow, to be executed only after the "Code capabilities" section of
`ablation-study.md` is implemented (refactor-plan Phase 2/3, owned by
another batch).

## Order of operations

1. **Metrics fix first** (`research-rules/metrics.md`): decide and
   implement macro-F1 reporting (either add it alongside weighted F1 in
   `Data.calc_metrics_sklearn`, or build the offline
   `predictions.csv → macro F1` recomputation script) before running or
   reporting any ablation number — otherwise the three sub-studies are not
   comparable to each other or to the paper's stated metric.
2. **§6.1 Representation ablation** — requires only the `embedding_variant`
   slicing capability (no new strategy class, no new provider). Lowest
   implementation cost; do this first.
   ```
   # illustrative CLI once embedding_variant is a params-JSON/CLI option:
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_representation.json --dataset_name=DANINHAS \
     --strategy_name SSRAEKmeansHCSampling --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results=results/ablation_representation/
   ```
   with `params_ablation_representation.json`'s `DANINHAS.embedding_variant`
   set to `spatial`, `spectral`, or `full` per run (3 variants × 3 seeds = 9
   runs). Reuse the existing `Q=13` SSRAE cache — slicing is a
   post-hoc operation on the already-cached full embedding, so **no new
   feature-extraction pass is needed** if the cache-keying fix from
   `research-rules/reproducibility.md` is in place.
3. **§6.2 Hierarchy ablation** — requires the config-driven hierarchy fix
   (remove the hardcoded `'DANINHAS'` key in
   `core/query_strategies/ssl_ssrae_sampling.py:66`) plus resolving the
   `sample_sizes`-derivation TBD in `ablation-study.md` §6.2. 5 configs
   (`L=1,2a,2b,3,4`) × 3 seeds = 15 runs, `n_query=100` fixed, SSRAE full
   embedding.
   ```
   CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
     --params_json params_ablation_hierarchy_<config_id>.json --dataset_name=DANINHAS \
     --strategy_name SSRAEKmeansHCSampling --n_query 100 --seed <1|2|3> \
     --n_round 8 --dir_results=results/ablation_hierarchy/
   ```
   one params JSON per `config_kmh` row in the ablation table (or a single
   params JSON with a CLI override for `config_kmh`, if the config layer
   supports it — TBD depends on refactor Phase 2 design).
4. **§6.3 Stage-contribution ablation** — highest implementation cost: needs
   the ImageNet-ResNet embedding provider (new) and the proportional-random
   flat k-means strategy (new class, or a config-driven variant of
   `SSRAEKmeansSampling` with the seed bug fixed — see
   `research-rules/reproducibility.md`). "RNHAL (full)" row needs **no new
   runs** — reuse `results/dalmax{1,2}/.../SSRAEKmeansHCSampling/` at
   `n_query=100`. The other two rows need new runs, all 3 seeds, at the
   `n_query` value(s) chosen for this study (TBD, see `n_query` sweep
   question flagged in `ablation-study.md` §6.1 — likely `n_query=100` for
   consistency with §6.1/§6.2, since the source runs for "RNHAL (full)" are
   available at that budget).

## Verification before trusting results

Before treating any ablation run's F1 as final, an `experiment-auditor`-style
pass (per `.claude/agents/experiment-auditor.md`, owned by another batch)
should check: (a) the SSRAE/embedding cache actually reflects the intended
`Q`/`embedding_variant`/dataset combination and not a stale cache, (b) the
`config_kmh` used matches the intended row of the §6.2 table exactly
(`n_levels`, `n_clusters`, `sample_sizes`), (c) the seed was propagated to
every source of randomness including k-means (see the hardcoded-seed issue
in `research-rules/reproducibility.md`), and (d) macro F1 (not the raw
`all_f1_score` weighted value) is what gets reported.

## Aggregation and paper hand-off

Once the runs for a given sub-study are complete, follow
`use-cases/generate-report.md` to aggregate across seeds, then map the
resulting table to the correct paper location per
`experiments/ablation-study.md`'s "Mapping to the paper's `\subsubsection`s"
table.
