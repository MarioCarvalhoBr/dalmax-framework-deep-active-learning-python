# Ablation params files

One params JSON per run-table row of `.specs/experiments/ablation-study.md`, in the Phase 2
`"embedding"`/`"selection"` config syntax (`dalmax.config.loader.load_experiment_config`). All
11 files here share `n_classes: 5`, `data_dir: "DATA/daninhas_full/"`, `n_epoch: 10`,
`train_args`/`test_args` batch size 256, `num_workers: 4` — identical to `params_df_gpu_0.json`'s
`DANINHAS` block, so the only per-file differences are the `embedding`/`selection` blocks. Every
file loads via `dalmax.config.loader.load_experiment_config`; see `tests/test_ablation_configs.py`.

`micro/` mirrors every file 1:1 (same 11 basenames) pointed at `DATA/daninhas_micro/` for local CPU
smoke-testing — see that subfolder's own note below and `scripts/ablations/smoke_ablations.sh`.

## §6.1 Representation ablation — `rep_full.json` / `rep_spatial.json` / `rep_spectral.json`

`embedding.variant` is `full` / `spatial` / `spectral` respectively (SSRAE, `Q=13`); `selection` is
held fixed at the **reference hierarchy** — `params_df_gpu_0.json`'s `config_kmh`
(`n_clusters=[600,200,100]`, `n_levels=3`, `sample_sizes=[30,15,2]`), exactly as
`ablation-study.md` §6.1's exact-config note specifies ("use the `run_pipe_gpu_0.sh` reference...
as done in the exact config below").

## §6.2 Hierarchy ablation — `hier_L1.json` .. `hier_L4.json`

`embedding` fixed at SSRAE full (`Q=13`); `selection.hierarchy` varies per
`ablation-study.md` §6.2's run table:

| File | n_clusters | n_levels | sample_sizes |
|---|---|---|---|
| `hier_L1.json` | `[50]` | 1 | `[3]` |
| `hier_L2a.json` | `[300, 100]` | 2 | `[15, 5]` |
| `hier_L2b.json` | `[100, 50]` | 2 | `[5, 3]` |
| `hier_L3.json` | `[300, 100, 50]` | 3 | `[15, 5, 3]` |
| `hier_L4.json` | `[300, 100, 50, 25]` | 4 | `[15, 5, 3, 2]` |

`n_query=100` for every row (CLI arg, not part of the JSON) per the spec.

**`sample_sizes` derivation rule** (documented per `ablation-study.md` §6.2's TBD, not yet
advisor-confirmed — see that file for the full semantics: `sample_sizes[level]` only affects
centroid-refinement resampling quality inside `HierarchicalKMeansSelection`, never the number of
ids `n_query` returns): `sample_sizes[i] ≈ round(n_clusters[i] * 0.05)`, floor 2. Verified against
the two on-disk reference configs' ratios (`params_df_gpu_0.json`: `30/600=0.05`, `15/200=0.075`,
`2/100=0.02`; `params_df_gpu_1.json`: `60/500=0.12`, `30/200=0.15`, `2/150≈0.013`) — the ~0.05 ratio
with a floor of 2 is the convention this ablation's five rows above follow consistently.
`dalmax.config.schema.HierarchyConfig` does not accept an unknown `_comment` key at its own level
(it only reads `n_clusters`/`n_levels`/`sample_sizes`), but `dalmax.config.loader._build_hierarchy_config`
tolerates (ignores) extra dict keys before constructing it — verified empirically and by
`tests/test_ablation_configs.py` — so `micro/*.json`'s hierarchy blocks carry an inline `_comment`
key safely; the full-scale files above keep the rule documented here instead, to match
`ablation-study.md`'s own JSON snippets byte-for-byte.

## §6.3 Stage contribution — `stage_full.json` / `stage_no_representation.json` / `stage_no_hierarchy.json`

- `stage_full.json`: SSRAE full (`Q=13`) + hierarchical, same reference hierarchy as §6.1
  (`[600,200,100]`/`[30,15,2]`) — the RNHAL-full config in the new config syntax. Per
  `ablation-study.md` §6.3, this row's *reported* macro F1 for the paper still comes from the
  already-executed `results/dalmax{1,2}/.../SSRAEKmeansHCSampling/` reference runs (recomputed
  offline for macro F1, see that file's "Metrics discrepancy" note) — this file exists so the row
  can also be re-run end-to-end through the new `RepresentationStrategy` pipeline for
  cross-checking, not because a re-run is required before the ablation can close out.
- `stage_no_representation.json`: `resnet_imagenet` (`q: null`, fixed 2048-d penultimate layer) +
  hierarchical, same reference hierarchy.
- `stage_no_hierarchy.json`: SSRAE full (`Q=13`) + `flat_proportional` (no `hierarchy` key — required
  by `dalmax.config.schema.SelectionConfig`, which rejects a `hierarchy` block for any
  non-`"hierarchical"` method).

## `micro/`

Same 11 files, mirrored for `make smoke-ablations` / `scripts/ablations/smoke_ablations.sh`:
`data_dir` -> `DATA/daninhas_micro/`, `n_classes: 5`, `n_epoch: 1`, batch size 16, `num_workers: 0`.

**2026-08-23 redefinition**: `DATA/daninhas_micro/` was redefined from a hand-picked 2-class/70-image
subset to a genuine **10%-stratified replica of `DATA/daninhas_full/`** — all 5 classes, both splits
(`scripts/make_micro_dataset.py`, see `.specs/adr/0004-micro-dataset-and-golden-run.md`'s amendment) —
so the local no-GPU smoke path exercises the same class count/imbalance shape as the real dataset
before a lab/Colab run, not just a 2-class toy case. That gives an unlabeled pool of **~796 images**
(806 train - 10 initial labeled, per the smoke script's `--n_init_labeled 10`) instead of the old
~40-image pool — almost exactly **1/10th** of the full-scale ablation pool (`~7986`, `8086` train -
`100` initial labeled), since the micro dataset *is* a 10% stratified sample. Hierarchies below are
therefore derived by dividing each full-scale `n_clusters` entry by 10 (rounding, floor 2), which
keeps the same relative shape as the full-scale row it mirrors:

| File | n_clusters | n_levels | sample_sizes |
|---|---|---|---|
| `hier_L1.json` | `[5]` | 1 | `[2]` |
| `hier_L2a.json` | `[30, 10]` | 2 | `[2, 2]` |
| `hier_L2b.json` | `[10, 5]` | 2 | `[2, 2]` |
| `hier_L3.json` | `[30, 10, 5]` | 3 | `[2, 2, 2]` |
| `hier_L4.json` | `[30, 10, 5, 2]` | 4 | `[2, 2, 2, 2]` |
| `rep_*.json` / `stage_full.json` / `stage_no_representation.json` (reference hierarchy) | `[60, 20, 10]` | 3 | `[3, 2, 2]` |

`sample_sizes` follow the same `~round(n_clusters[i] * 0.05)`, floor 2 rule as the full-scale files
(§6.2 above); at this small scale the floor dominates every entry except the reference hierarchy's
first level (`round(60*0.05)=3`).

All 11 configs were re-verified end-to-end against the real, regenerated `DATA/daninhas_micro/`
(`bash scripts/ablations/smoke_ablations.sh`, `--seed 1 --n_init_labeled 10 --n_query 5 --n_round 1
--device cpu`) after this redefinition: **11/11 passed**, including `hier_L2a.json`'s `[30, 10]` (each
of the 10 level-2 super-clusters would receive exactly 3 level-1 sub-clusters on a perfectly even
split — the same *shape* of risk documented below for the old `hier_L2b.json` — but the real SSRAE
embedding over the new ~796-image pool did not produce an exactly-even split, so the vendored
`dtype=object` equal-subcluster-size bug (`dalmax/tools/SSL/src/utils.py:28`, see
`dalmax/selection/hierarchical_kmeans.py`'s module docstring) was not triggered). Total wall time for
all 11 configs: **~4m37s** on the local CPU-only dev notebook, under the ~5 min target. This is an
empirical, seed/dataset-specific result, not a structural guarantee — as with the pre-redefinition
`hier_L2b.json` note below, an exactly-even k-means split is possible in principle for a different
seed and would need to be re-probed if the sampling seed or dataset ever changes again.

`hier_L4.json`'s micro hierarchy deliberately avoids ever reaching a final cluster of size 1: the
vendored `core/tools/SSL/src/hierarchical_kmeans_gpu.py`/`hierarchical_sampling.py` pipeline this
wraps (`dalmax/selection/hierarchical_kmeans.py`) has no documented/tested behavior for
`n_clusters=1` at the deepest level, so `[30, 10, 5, 2]` (bottoming out at 2, not 1) is used instead
of a naive halving-to-1 progression — a deliberately conservative choice, not a verified requirement;
`tests/test_ablation_configs.py` only checks that the config *loads*, not that a real hierarchical
selection run over it succeeds (that is exercised by `scripts/ablations/smoke_ablations.sh` instead,
over the real micro dataset).

**Historical note (pre-2026-08-23-redefinition `hier_L2b.json`'s `[5, 2]`, not the naive `[4, 2]`
halving of `hier_L2a`'s `[8, 4]`)**: on the old 2-class/70-image micro dataset, `[4, 2]` triggered the
pre-existing vendored `dtype=object` equal-subcluster-size bug documented above — empirically, at
`--seed 1` over that dataset's real SSRAE-full embedding, 4 level-1 clusters split exactly 2-and-2
into the 2 level-2 super-clusters, hitting the bug; `[5, 2]` did not split evenly and was verified to
run cleanly. This finding is dataset/seed-specific and does not directly carry over to the
2026-08-23-redefined 5-class/806-train-image dataset (a completely different embedding), which is why
`hier_L2b.json` was independently re-probed above (`[10, 5]`, passing) rather than reusing the old
ratio unverified. The vendored bug itself is not fixed (out of scope, per
`.claude/rules/data-safety.md`'s "wrap, don't edit, the vendored SSL code" policy); the full-scale
`hier_L2b.json` (`[100, 50]`, ~8k-image pool) is not affected in practice, since an exact-even split
at that scale is astronomically unlikely.
