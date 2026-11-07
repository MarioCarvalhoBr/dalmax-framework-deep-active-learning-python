# ADR 0012: KMH parameter sweep at every paper-1 budget

- **Status:** Accepted (author decision 2026-11-07; prepared, not executed)
- **Date:** 2026-11-07

## Context

Paper 1 benchmarks 12 classical query strategies plus KMH (hierarchical k-means over frozen
ImageNet ResNet-50 features) at `n_query` 10, 50 and 100 (`.specs/experiments/campaign.md`). KMH
runs with one fixed configuration at every budget: the RNHAL reference hierarchy `[600,200,100]`
with resampling sizes `[30,15,2]` (`files_config/campaign/params_kmh.json`), k-means++
initialisation, closest-to-centroid resampling over 10 iterations and random draws inside the leaf
clusters. None of these was chosen for paper 1; they were inherited from the RNHAL reference.

Two facts motivate a sweep. First, the budget is distributed over the coarsest level, which has
100 clusters: at `n_query=100` every cluster receives one image, but at `n_query=50` only 50
randomly chosen clusters do and at `n_query=10` only 10, so at small budgets KMH is largely a draw
of clusters and the lower levels barely act. Second, the executed campaign shows KMH winning at
`n_query=10` (0.6466 weighted F1, next 0.5357), within the seed deviation of the winner at
`n_query=50` (0.7833 vs Adversarial DeepFool 0.7913, at 30 times less query time) and 0.049
below Least Confidence at `n_query=100` (0.8224 vs 0.8716), while the RNHAL hierarchy ablation
moved the result by up to 7 points between hierarchies (0.8287 to 0.8955). The author's goal
(2026-11-07) is to find the strongest KMH configuration per budget, decide from the results
whether the paper-1 reference changes, and give paper 1 a KMH ablation; papers 2 and 3 would then
build on the strongest KMH.

## Decision

We will run a KMH parameter sweep as a separate manifest (`files_config/campaign/manifest_kmh_sweep.json`,
results under `results/campaign_kmh/`, parts `kmh_nq10`, `kmh_nq50`, `kmh_nq100`), varying every
knob the hierarchical k-means exposes, at the three paper-1 budgets, seeds 1-3, the paper-1
protocol otherwise unchanged (`n_init_labeled=100`, `n_round=8`, 10 epochs):

1. **Hierarchy axis** (nine shapes per budget, other knobs at their defaults): `L=1` with
   `[n_q]`, `[100]`, `[200]`, `[600]`; `L=2` with `[200,100]` and the budget-matched `[6n_q, n_q]`;
   `L=3` reference `[600,200,100]` and the budget-matched `[6n_q, 2n_q, n_q]`; `L=4`
   `[800,600,200,100]`. The budget-matched shapes put `n_q` clusters at the coarsest level so that
   every coarse cluster contributes one image whatever the budget. At `n_query=100` the matched
   shapes coincide with `[100]` and the reference (aliases, not runs).
2. **Knob axis**, applied to the reference and to the budget-matched `L=3` hierarchy: k-means
   initialisation `random` instead of `kmeans++`; `closest`-to-centroid draws inside the leaves
   instead of random; no resampling (`n_resamples=0`) instead of 10 iterations; random instead
   of closest-to-centroid resampling.
3. The four knobs become params-JSON keys of `selection.hierarchy` (`n_resamples`, `init_method`,
   `resample_strategy`, `leaf_sampling`) with defaults equal to the previous hard-wired behaviour,
   so every executed run and every existing params file keeps its signature.
4. The reference hierarchy is **re-run inside the sweep** at each budget instead of being aliased to
   the campaign runs, so that every comparison of the sweep is made within one runtime image
   (KI-43, `.specs/quality/known-issues.md`); the sweep's reference rows double as a KI-43 control
   for KMH. The executed campaign stays untouched (`results/` is append-only).
5. Resampling sizes of new hierarchies follow the rule `max(2, round(0.05 * k))` per level (the
   reference's ratio 30/600, already used by `files_config/ablations/rnhal/hier_L4_k800_600_200_100.json`);
   the reference keeps its historical `[30,15,2]`.

Totals: 45 run groups, 135 runs (51 + 51 + 33). The sweep is the first stage; the grid is data at
the top of `scripts/campaign/build_kmh_sweep_configs.py` so that a second stage (for example the
knobs on more hierarchies, `num_init`, or the same knobs on RNHAL and TexHAL) is a list edit.

## Consequences

- Paper 1 gains a KMH sensitivity study; its main tables keep the executed reference until the
  author decides, from the sweep tables, whether the reference changes (`thesis_drafts/PENDENCIAS.md`
  P17). If it changes, the shared `shared/kmh_nq100` run ("w/o representation module" of papers 2
  and 3) stays as executed, and the papers must say which KMH configuration each table uses.
- The selection of the best configuration is made on the test set (there is no validation split in
  the protocol); the papers must disclose it, as they already do for the RNHAL `L=4` hierarchy.
- Cost: about 135 runs of 8 rounds each; the reference costs about 64 s of clustering per round,
  the `L=1` shapes much less, so roughly 12 to 18 GPU hours on Colab, split by budget across
  sessions. Each run writes a ResNet-50 checkpoint (about 95 MB): about 13 GB on Drive.
- The config schema change (four optional keys with defaults) is recorded in
  `.specs/experiments/experimental-protocol.md` and `.specs/architecture/current-state.md`; the
  sweep protocol in `.specs/experiments/campaign.md`.
