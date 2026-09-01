# Ablation params files

This folder holds one params JSON per ablation run-table row, organized **by paper/method**
since 2026-08-30 (see `.specs/experiments/papers-roadmap.md` for the three-paper plan this
folder structure reflects):

```
files_config/ablations/
├── README.md          # this file
├── rnhal/              # paper 3 — RNHAL (SSRAE + hierarchical k-means), ablations EXECUTED
│   ├── rep_full.json / rep_spatial.json / rep_spectral.json   (§6.1)
│   ├── hier_L1.json .. hier_L4.json                            (§6.2)
│   ├── stage_full.json / stage_no_representation.json / stage_no_hierarchy.json  (§6.3)
│   └── micro/           # CPU smoke mirrors of the 11 files above
└── texhal/              # paper 2 — TexHAL (VCTex + hierarchical k-means), ablations TBD
    ├── rep_q5.json / rep_q17.json / rep_full.json               (§6.1)
    ├── hier_L1.json .. hier_L4.json                              (§6.2)
    ├── stage_full.json / stage_no_representation.json / stage_no_hierarchy.json  (§6.3)
    └── micro/           # CPU smoke mirrors of the 11 files above
```

Each method's `.json` files load via `dalmax.config.loader.load_experiment_config`, using the
Phase 2 `"embedding"`/`"selection"` config syntax; see `tests/test_ablation_configs.py`
(parametrized over both `rnhal/` and `texhal/`, 44 files total: 11 full-scale + 11 micro per
method).

## Three-paper mapping (see `.specs/experiments/papers-roadmap.md` for the full plan)

| Paper | Method | CLI preset (unchanged) | Ablation suite | Status |
|---|---|---|---|---|
| 1 — DAL benchmark comparison | all non-representation baselines (Random, Entropy, Margin, BALD, ...) | n/a | none (no ablations for paper 1) | n/a |
| 2 — TexHAL proposal | VCTex + hierarchical k-means | `VCTexKmeansHCSampling` (or generic `RepresentationStrategy` with `embedding.extractor="vctex"`) | `files_config/ablations/texhal/` | configs materialized, not yet run |
| 3 — RNHAL proposal | SSRAE + hierarchical k-means | `SSRAEKmeansHCSampling` (or generic `RepresentationStrategy` with `embedding.extractor="ssrae"`) | `files_config/ablations/rnhal/` | **executed** 2026-08-26 on Colab, see `.specs/experiments/ablation-study.md` |

Every file in both method folders uses the generic `RepresentationStrategy` `--strategy_name`
(never the fixed presets) so the params JSON's `"embedding"`/`"selection"` blocks are actually
read — see `.specs/experiments/ablation-study.md` §6.1's note on why the preset ignores those
blocks entirely.

## Naming convention (shared by both methods)

Both method folders mirror the same 11 basenames so the two suites stay structurally comparable
(the same GPU-split scripts, same reporting tool, same study grouping):

| Study | RNHAL basenames | TexHAL basenames | Why the names differ |
|---|---|---|---|
| §6.1 representation | `rep_full`, `rep_spatial`, `rep_spectral` | `rep_q5`, `rep_q17`, `rep_full` | RNHAL's §6.1 slices one SSRAE embedding into spatial/spectral column-groups (`embedding.variant`); TexHAL has no spatial/spectral split — `slice_embedding` is SSRAE-only — so its §6.1 instead sweeps VCTex's own multi-scale hyperparameter `Q` (single-scale `Q=5`, single-scale `Q=17`, and the paper's multi-scale `Q=[5,17]` "full") |
| §6.2 hierarchy | `hier_L1` .. `hier_L4` | `hier_L1` .. `hier_L4` | identical k-grids/sample_sizes, only the embedding block differs (vctex q=[5,17] vs ssrae q=13) |
| §6.3 stage contribution | `stage_full`, `stage_no_representation`, `stage_no_hierarchy` | `stage_full`, `stage_no_representation`, `stage_no_hierarchy` | identical structure; `stage_no_representation` is byte-for-byte identical between the two methods (see below) |

`variant` is always `"full"` for every `texhal/*.json` file — `dalmax/embeddings/variants.py::
slice_embedding` (the `full`/`spatial`/`spectral` column-group slicer) only applies when
`embedding_provider.name == "ssrae"` (`dalmax/query_strategies/representation.py`); setting
`variant` to anything but `"full"` for `extractor: "vctex"` would be silently ignored by the
pipeline, so it is never done here.

## §6.1 — representation ablation

- **RNHAL** (`rnhal/rep_full.json` / `rep_spatial.json` / `rep_spectral.json`): fixed SSRAE `Q=13`,
  `embedding.variant` swept over `full`/`spatial`/`spectral`; `selection` held at the **reference
  hierarchy** (`n_clusters=[600,200,100]`, `n_levels=3`, `sample_sizes=[30,15,2]`, matching
  `params_df_gpu_0.json`'s `config_kmh`).
- **TexHAL** (`texhal/rep_q5.json` / `rep_q17.json` / `rep_full.json`): VCTex's `Q` is a *tuple of
  scales* (the paper's best-parameters value is `Q=(5,17)`, i.e. two RNN patch scales
  concatenated), not a single spatial/spectral-block split — so this ablation instead isolates
  each individual scale (`q: [5]`, `q: [17]`) against the multi-scale combination (`q: [5, 17]`,
  named `rep_full.json` for structural parity with RNHAL's "Full" row). `selection` is held at the
  **same reference hierarchy** as RNHAL's §6.1 (`[600,200,100]`/`[30,15,2]`) for cross-paper
  comparability (confirmed 2026-09-01).

## §6.2 — hierarchy ablation

Identical `n_clusters`/`n_levels`/`sample_sizes` grid for both methods (see the table in each
method's own historical README section below, or `.specs/experiments/ablation-study.md` §6.2 /
`ablation-study-texhal.md` §6.2) — only `embedding` differs: RNHAL fixes SSRAE full (`Q=13`),
TexHAL fixes VCTex multi-scale (`q: [5, 17]`, `variant: "full"`).

## §6.3 — stage contribution

- `stage_full.json`: the method's own representation (SSRAE full for RNHAL, VCTex `q=[5,17]` for
  TexHAL) + hierarchical selection, reference hierarchy.
- `stage_no_representation.json`: `resnet_imagenet` (`q: null`, fixed 2048-d penultimate layer) +
  hierarchical, reference hierarchy — **byte-for-byte identical between `rnhal/` and `texhal/`**,
  kept as two separate files (not a shared symlink) for batch-internal consistency — each method's
  GPU scripts and results tree reference their own copy under `results/ablations/{method}/6_3/
  stage_no_representation/`. Since neither method's representation module is involved in this row,
  its result should numerically match the other method's `stage_no_representation` run within
  ordinary GPU/cuDNN run-to-run variance (`.claude/rules/reproducibility.md`) — this is documented
  as a useful sanity cross-check in `.specs/experiments/ablation-study-texhal.md`, not enforced by
  any test.
- `stage_no_hierarchy.json`: the method's own representation + `flat_proportional` (no `hierarchy`
  key — required by `dalmax.config.schema.SelectionConfig`, which rejects a `hierarchy` block for
  any non-`"hierarchical"` method).

## `micro/` (both methods)

Same 11 basenames per method, mirrored for `make smoke-ablations` /
`scripts/ablations/smoke_ablations.sh`: `data_dir` -> `DATA/daninhas_micro/`, `n_classes: 5`,
`n_epoch: 1`, batch size 16, `num_workers: 0`. TexHAL's micro hierarchies are copied verbatim from
RNHAL's micro hierarchy choices (same `n_clusters`/`sample_sizes` values — see
`rnhal/micro/README` derivation below), since both suites share the same ~796-image
`DATA/daninhas_micro/` unlabeled pool and the same k-means machinery; only the embedding block
differs (vctex `q=[5,17]` vs ssrae `q=13`, per method).

`make smoke-ablations` was the **first real execution of VCTex through the generic
`RepresentationStrategy` path** — previously VCTex only ran through the legacy
`VCTexKmeansSampling`/`VCTexKmeansHCSampling` presets. Result (2026-08-30): **12/12 passed, no
`VCTexProvider` bug found** — see `.specs/experiments/ablation-study-texhal.md`'s
"VCTex-through-generic-path findings" section for the full timing and the empirically-confirmed
embedding-dimension table (`27*(Q+1)` per scale).

---

## RNHAL micro-hierarchy derivation (historical, unchanged by this reorganization)

The **2026-08-23 redefinition**: `DATA/daninhas_micro/` was redefined from a hand-picked 2-class/
70-image subset to a genuine **10%-stratified replica of `DATA/daninhas_full/`** — all 5 classes,
both splits (`scripts/make_micro_dataset.py`, see `.specs/adr/0004-micro-dataset-and-golden-run.md`'s
amendment) — so the local no-GPU smoke path exercises the same class count/imbalance shape as the
real dataset before a lab/Colab run, not just a 2-class toy case. That gives an unlabeled pool of
**~796 images** (806 train - 10 initial labeled, per the smoke script's `--n_init_labeled 10`)
instead of the old ~40-image pool — almost exactly **1/10th** of the full-scale ablation pool
(`~7986`, `8086` train - `100` initial labeled), since the micro dataset *is* a 10% stratified
sample. Hierarchies are therefore derived by dividing each full-scale `n_clusters` entry by 10
(rounding, floor 2), which keeps the same relative shape as the full-scale row it mirrors:

| File | n_clusters | n_levels | sample_sizes |
|---|---|---|---|
| `hier_L1.json` | `[5]` | 1 | `[2]` |
| `hier_L2a.json` | `[30, 10]` | 2 | `[2, 2]` |
| `hier_L2b.json` | `[10, 5]` | 2 | `[2, 2]` |
| `hier_L3.json` | `[30, 10, 5]` | 3 | `[2, 2, 2]` |
| `hier_L4.json` | `[30, 10, 5, 2]` | 4 | `[2, 2, 2, 2]` |
| `rep_*.json` / `stage_full.json` / `stage_no_representation.json` (reference hierarchy) | `[60, 20, 10]` | 3 | `[3, 2, 2]` |

`sample_sizes` follow the `~round(n_clusters[i] * 0.05)`, floor 2 rule verified against the two
on-disk `params_df_gpu_{0,1}.json` reference configs' ratios; see the git history of this file
(pre-2026-08-30 reorganization) for the full historical derivation notes, equal-subcluster-size
bug investigation (`dalmax/tools/SSL/src/utils.py:28`), and the `hier_L2b.json`/`hier_L4.json`
per-config rationale — all of that content is unchanged by this reorganization, only its location
moved from this file into `rnhal/`'s config files themselves (the full-scale files still carry
the derivation rule in prose here since it applies to both methods equally).

`sample_sizes` derivation rule (applies to both `rnhal/` and `texhal/`): `sample_sizes[i] ≈
round(n_clusters[i] * 0.05)`, floor 2. `dalmax.config.schema.HierarchyConfig` does not accept an
unknown `_comment` key at its own level, but `dalmax.config.loader._build_hierarchy_config`
tolerates (ignores) extra dict keys before constructing it — verified empirically and by
`tests/test_ablation_configs.py` — so `*/micro/*.json`'s hierarchy blocks carry an inline
`_comment` key safely; the full-scale files keep the rule documented here instead.

All 44 configs (22 per method) load via `dalmax.config.loader.load_experiment_config` —
`tests/test_ablation_configs.py`. `bash scripts/ablations/smoke_ablations.sh` (default: both
methods) end-to-end smoke-tests every config against the real, regenerated
`DATA/daninhas_micro/`.
