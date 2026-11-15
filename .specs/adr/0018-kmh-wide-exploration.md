# ADR 0018: Explore wider KMH hierarchies (part `kmh_wide` of the KMH sweep, account B)

- **Status:** Accepted (author decision 2026-11-15); prepared, not executed
- **Date:** 2026-11-15

## Context

Paper 1's KMH is hierarchical k-means selection on frozen ImageNet ResNet-50 features (`RepresentationStrategy`
with `embedding.extractor = "resnet_imagenet"`). On the clean campaign (ADR 0016, `DATA/daninhas_full_fixed`,
462 runs executed and verified locally: OK=462, one commit, seed and data-order audits PASS), final-round weighted
F1, mean of seeds 1-3, KMH is below the best benchmark method, WAAL, at every query size:

| n_query | best KMH configuration of the sweep | WAAL | gap |
|---|---|---|---|
| 100 | `kmh/nq100/hier_L3_ref` `k=[600,200,100]` 0.7921 | 0.8077 | 1.6 points |
| 50 | `kmh/nq50/hier_L3_ref__init_random` 0.7146 | 0.7874 | 7.3 points |
| 10 | `kmh/nq10/hier_L1_k600` `k=[600]` 0.5627 | 0.5793 | 1.7 points |

The 55 executed groups of the KMH sweep (ADR 0012, `results/campaign_kmh_fixed/`) point in two directions:

- At nq100 the best shapes have a wide first level and a last level equal to the query size:
  `[600,200,100]` 0.7921, `[800,600,100]` 0.7740, `[600,100]` 0.7669 (hierarchy axis, default knobs).
- At nq10 the widest first levels are best (`[600]` 0.5627, `[800,600,200,100]` 0.5216), and the
  "budget-matched" shapes that shrink every level with the budget are among the worst (`[60,20,10]` 0.4043,
  `[60,10]` 0.4109).
- The wide-top shape whose last level equals the query size was never run at nq10 and nq50 (the reference
  `[600,200,100]` is used there unchanged), and no first level wider than 1,200 clusters was ever run.

The seed spread is large: the standard deviation over the three seeds is 5 to 10 points for most configurations
at nq50 and nq10 (e.g. `hier_L1_k600` at nq50: 0.6199 +- 0.1351).

## Decision

We will run an exploration round of these two directions as a new part **`kmh_wide`** of the KMH sweep manifest
(`files_config/campaign/manifest_kmh_sweep.json`, generator `scripts/campaign/build_manifest.py`; configs from
`scripts/campaign/build_kmh_sweep_configs.py::WIDE_POINTS`): 13 run groups x seeds 1-3 = 39 runs,
`RepresentationStrategy` with ImageNet features, NIL 100, 8 rounds, every knob other than the hierarchy shape at
the sweep's defaults (k-means++ init, `n_resamples` 10, resample `closest`, leaf `random`).

| group id | `n_clusters` | `sample_sizes` | direction |
|---|---|---|---|
| `kmh_wide/nq100/hier_L1_k1200` | `[1200]` | `[60]` | (b) |
| `kmh_wide/nq100/hier_L1_k2400` | `[2400]` | `[120]` | (b) |
| `kmh_wide/nq100/hier_L3_k1200_400_100` | `[1200,400,100]` | `[60,20,5]` | (b) |
| `kmh_wide/nq50/hier_L3_k600_200_50` | `[600,200,50]` | `[30,10,3]` | (a) |
| `kmh_wide/nq50/hier_L2_k600_50` | `[600,50]` | `[30,3]` | (a) |
| `kmh_wide/nq50/hier_L1_k1200` | `[1200]` | `[60]` | (b) |
| `kmh_wide/nq50/hier_L1_k2400` | `[2400]` | `[120]` | (b) |
| `kmh_wide/nq50/hier_L3_k1200_400_50` | `[1200,400,50]` | `[60,20,3]` | (b) |
| `kmh_wide/nq10/hier_L3_k600_200_10` | `[600,200,10]` | `[30,10,2]` | (a) |
| `kmh_wide/nq10/hier_L2_k600_10` | `[600,10]` | `[30,2]` | (a) |
| `kmh_wide/nq10/hier_L1_k1200` | `[1200]` | `[60]` | (b) |
| `kmh_wide/nq10/hier_L1_k2400` | `[2400]` | `[120]` | (b) |
| `kmh_wide/nq10/hier_L3_k1200_400_10` | `[1200,400,10]` | `[60,20,2]` | (b) |

Direction (a): wide top with the last level equal to the query size; direction (b): a wider top.

- **Sample sizes** follow the rule every non-reference sweep config was generated with
  (`build_kmh_sweep_configs.py::sample_sizes_for`): `sample_sizes[i] = max(2, round_half_up(0.05 * n_clusters[i]))`
  (so `k=50` gives 3, `k=10` gives 2); only the reference keeps its historical `[30,15,2]`.
- **Params** under `files_config/ablations/kmh_sweep/nq{N}/<config>.json` (micro mirrors under `micro/`, clusters
  / 10, minimum 2), every earlier file byte-identical. None has the effective signature of an executed group or of
  a `texhal_q13` group (`tests/test_campaign.py::test_kmh_wide_signatures_differ_from_every_v3_group`).
- **Layout**: group ids `kmh_wide/nq{N}/<config>`, results under
  `results/campaign_kmh_fixed/wide/nq{N}/<config>/`, a folder of its own so the notebook can zip the part alone.
  The groups are appended after the 55 executed sweep groups (nothing executed moves).
- **GPU part / stage**: all 13 groups in `part_2_gpu` (Colab account B), `gpu_stage` 3 (the sweep's stage).
  Account B's Drive already holds the ImageNet feature cache of seeds 1-3: the cache key
  (`dalmax/embeddings/cache.py`, built in `dalmax/query_strategies/representation.py::_compute_initial_pool_embeddings`)
  is `(dataset, extractor, Q, variant "full", split "train", order_hash, pool_hash)`, where `pool_hash` hashes the
  INITIAL unlabeled pool ids. That pool is fixed by the seed and `n_init_labeled` (100 everywhere) only, so the key
  depends neither on `n_query` nor on the hierarchy: the nq10 sweep groups of account B wrote exactly the three
  files the 39 new runs will read (the seed audit confirms one initial set per seed across all 462 executed runs).
- **GPU time estimates** (`files_config/campaign/gpu_time_estimates.json`, `basis: estimate`, rule in
  `scripts/campaign/build_gpu_time_estimates.py::fit_estimate`): at the same query size and number of levels, the
  least-squares line of the measured seconds per run of the sweep groups with default knobs against the total
  number of clusters `sum(n_clusters)`, evaluated at the group's own total. Rationale: every Lloyd iteration of the
  vendored k-means recomputes the centroids in a Python loop over the clusters (`kmeans_gpu.compute_centroids`),
  at every level and resampling step, so the selection cost grows linearly with the total number of clusters; the
  measured groups follow such a line closely (about 0.54 s per cluster per run at every query size; at nq100 the
  L=3..5 shapes up to 2,700 clusters in total sit on it). Total 9.2 GPU h (per run: 714 to 1,380 s for the
  one-level shapes, 392 to 998 s for the others). The one-level `k=1200/2400` points extrapolate beyond the
  measured one-level range (`k <= 600`), so they are the least certain. Measured values are untouched.
- **Reports**: one table per query size, `kmh_wide_nq100`, `kmh_wide_nq50`, `kmh_wide_nq10` (paper1, only the new
  groups, so each table stays inside one GPU part); the ids and rows of every existing sweep table are unchanged.
- **Engine**: `dalmax/campaign.py` gains the semantic part name `kmh_wide` (`PARTS`, `PHASE_LABELS` after
  `kmh_nq10`), the same minimal way `texhal_q13` was added (ADR 0017). Nothing `tools/trainer.py` executes changes.
- **Thesis selection rule** (thesis side, not implemented here): the rule "paper 1's KMH row = the highest mean
  final-round weighted F1 of the KMH study at that query size" will now range over the `kmh_nq{N}` groups AND the
  `kmh_wide` groups of that size.

## Feasibility

Checked on the vendored code (`dalmax/tools/SSL/src/`) and by running it on CPU:

- Every shape has strictly fewer clusters at each level than at the level below, and the pool has 7,130 (round 1)
  to 6,430 (round 8) unlabeled images, so no level asks for more clusters than it has points
  (`kmeans_gpu.kmeans` would silently cap `n_clusters` at the number of points otherwise). The resampling of level
  1 with `sample_sizes` 60/120 takes up to 60/120 points per cluster, i.e. essentially every point when the mean
  cluster size is about 3 to 6, so the resampled k-means always has more points than clusters; the top levels of
  `[600,10]` and `[1200,400,10]` resample 2 points from each of 10 clusters (20 points for 10 clusters), the same
  situation as the executed `[60,20,10]` (sample size 2 at its last level).
- Empty clusters cannot break the run: k-means++ seeds every centroid on a data point, empty clusters keep their
  centroid in `compute_centroids`, and `hierarchical_sampling.py` gives an empty cluster a zero quota (KI-44 fix).
- The water-filling allocation (`find_subcluster_target_size`) gives one image to each of the `n_query` top-level
  clusters when the last level equals the query size (direction (a) and the three-level shapes of (b)), and for a
  single level with `k >> n_query` gives one image to each of `n_query` clusters drawn uniformly among the `k`
  (`np.random.choice` over the clusters larger than the cut 0), so at nq10 the one-level `k=2400` query is 10
  images from 10 of 2,400 clusters (about 3 images each). This is the same mechanism as the executed
  `hier_L1_k600` at nq10/nq50; it is close to a cluster-uniform random draw, which is part of what the round tests.
- CPU check (`make campaign-run MANIFEST=files_config/campaign/manifest_kmh_sweep_micro.json PART=kmh_wide SEEDS=1
  DEVICE=cpu`, the 13 micro mirrors, then `make campaign-verify` on them) and a full-size synthetic check of the 13
  full-scale shapes at 7,130 and 6,430 points: see the execution record in `.specs/experiments/campaign.md`
  ("KMH wide-top exploration").

## Selection on the test set

There is no validation split: every KMH configuration is compared by its final-round TEST weighted F1, and the
selected value is then reported on the same test set. Every extra configuration compared this way makes the
selected KMH value more optimistic (the maximum of more noisy estimates), and the seed spread here is 5 to 10
points, larger than the gaps to WAAL being chased. Therefore a configuration chosen from this round will be
**CONFIRMED on new seeds (4, 5, 6)**, run together with `RandomSampling` and `WAAL` at the same query size, before
any claim that KMH matches or beats WAAL. That confirmation part is planned, not built here (a later change: its
own part and results folder, the three methods x the chosen query size(s) x seeds 4-6). Until it exists, the thesis
reports the `kmh_wide` numbers as exploration, disclosed as selected on the test set, like ADR 0017's `Q=13`.

## Consequences

- KMH sweep manifest 204 runs / 68 groups (was 165 / 55); `part_2_gpu` 216 runs / 72 groups (was 177 / 59);
  `part_1_gpu` unchanged (309 / 103); both manifests 525 runs (was 486). The main manifest is byte-identical. The
  107 + 55 groups of tag `campaign-fixed-v3` keep id, parts, stage, params file, budgets, seeds, results directory
  and effective signature (`tests/test_campaign.py::test_v3_groups_keep_their_identity` pins a digest of the v3
  state, computed on a checkout of the tag; the earlier v2 digest test stays).
- The ADR 0016 balance of the two GPU parts no longer holds once the two add-on parts are counted (51.3 vs 58.0
  estimated GPU h, 109.3 in total); the balance test now checks it without `texhal_q13` and `kmh_wide`, which
  went to the account that already holds their caches.
- New tag **`campaign-fixed-v4`** (created by the author after review) is the notebook's `DEFAULT_GIT_REF`; its
  run-affecting code equals v2's and v3's (only `dalmax/campaign.py`'s part registry changed). On account B the
  code-version audit lists **two commits** (the v2 commit `464a5cc` for the 177 earlier leaves of `part_2_gpu`, the
  v4 commit for the 39 new ones), an expected WARN. Account A may still run `texhal_q13` with v3 or v4 (identical
  trainer code); the merged `make campaign-verify-all` then lists up to three commits (v2, v3, v4), all explained
  here and in ADR 0017.
- The notebook's `ZIP_ONLY_PART` gains `"kmh_wide"` (folder `results/campaign_kmh_fixed/wide`, 39 checkpoints, the
  sweep root's logs renamed `campaign_part_2_gpu__kmh_wide.log` / `failures_part_2_gpu__kmh_wide.log`);
  `"texhal_q13"` works as before. Operator steps: `COLAB_RUNBOOK.md` section 6f.
- Follow-up after execution: record the outcome here and in `.specs/experiments/campaign.md`; if a `kmh_wide`
  configuration becomes paper 1's KMH at some query size, build the confirmation part (seeds 4-6) first.
