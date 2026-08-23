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
`data_dir` -> `DATA/daninhas_micro/`, `n_classes: 2`, `n_epoch: 1`, batch size 16, `num_workers: 0`.
Hierarchies are scaled down for the micro dataset's ~40-image unlabeled pool (50 train images - 10
initial labeled, per the smoke script's `--n_init_labeled 10`):

| File | n_clusters | n_levels | sample_sizes |
|---|---|---|---|
| `hier_L1.json` | `[8]` | 1 | `[2]` |
| `hier_L2a.json` | `[8, 4]` | 2 | `[2, 1]` |
| `hier_L2b.json` | `[5, 2]` | 2 | `[1, 1]` |
| `hier_L3.json` | `[8, 4, 2]` | 3 | `[2, 1, 1]` |
| `hier_L4.json` | `[12, 6, 3, 2]` | 4 | `[3, 1, 1, 1]` |
| `rep_*.json` / `stage_full.json` / `stage_no_representation.json` (reference hierarchy) | `[8, 4, 2]` | 3 | `[2, 1, 1]` |

`hier_L4.json`'s micro hierarchy deliberately avoids ever reaching a final cluster of size 1: the
vendored `core/tools/SSL/src/hierarchical_kmeans_gpu.py`/`hierarchical_sampling.py` pipeline this
wraps (`dalmax/selection/hierarchical_kmeans.py`) has no documented/tested behavior for
`n_clusters=1` at the deepest level, so `[12, 6, 3, 2]` (bottoming out at 2, not 1) is used instead
of the naive halving-to-1 progression (`[8, 4, 2, 1]`) — a deliberately conservative choice, not a
verified requirement; `tests/test_ablation_configs.py` only checks that the config *loads*, not
that a real hierarchical selection run over it succeeds (that is exercised by
`scripts/ablations/smoke_ablations.sh` instead, over the real micro dataset).

**`hier_L2b.json`'s `[5, 2]` (not the naive `[4, 2]` halving of `hier_L2a`'s `[8, 4]`)**: `[4, 2]`
triggers the pre-existing vendored `dtype=object` equal-subcluster-size bug documented in
`dalmax/selection/hierarchical_kmeans.py`'s module docstring (`core/tools/SSL/src/utils.py:28`) —
empirically, at `--seed 1` over the real `DATA/daninhas_micro` SSRAE-full embedding, 4 level-1
clusters split exactly 2-and-2 into the 2 level-2 super-clusters, hitting the bug. `[5, 2]` does not
split evenly and was verified (by direct probing with the actual derived selection RNG, real
embeddings, `--seed 1`) to run cleanly; `[6,3]`/`[6,2]`/`[9,3]` were also probed and rejected for the
same reason, `[5,2]`/`[7,3]`/`[10,3]` all passed — `[5,2]` was chosen as the closest match to
`hier_L2a`'s `[8,4]` shape. This is a smoke-test-only, dataset/seed-specific workaround — the
vendored bug itself is not fixed (out of scope, per `.claude/rules/data-safety.md`'s "wrap, don't
edit, the vendored SSL code" policy) and could in principle resurface for a different seed; the
full-scale `hier_L2b.json` (`[100, 50]`, ~10k-image pool) is not affected in practice, since an
exact-even split at that scale is astronomically unlikely.
