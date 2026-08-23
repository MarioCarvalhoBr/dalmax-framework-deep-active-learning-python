---
name: ablation-study
description: Operational guide for the three RNHAL ablation studies (representation, hierarchy, stage contribution) - which code changes each requires, which configs to run, expected outputs, and how results feed the paper.
---

# Ablation study

This mirrors `.specs/experiments/ablation-study.md` — the advisor-requested
ablation section for the paper's `\subsection{Ablation study}`. All three
sub-studies are evaluated with **macro F1** on the held-out split, dataset
`daninhas_full`, using the seeds and budgets from
`.specs/experiments/experimental-protocol.md` (seeds 1-3, matching
`run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`).

**Status (2026-08-23, Phase 2 landed): runnable via config today, not yet run.** Every "Not
implemented" note below is now implemented — `dalmax/embeddings/`, `dalmax/selection/`, the generic
`RepresentationStrategy` CLI name, and macro-F1 in `results.json` all exist and are tested. See
`.specs/experiments/ablation-study.md` for the exact params-JSON snippets and CLI invocations for
every row, and `.specs/use-cases/run-ablation.md` for the operational run order. This file's content
below is kept as the conceptual/background reference (embedding layout, what each ablation isolates)
— for "what to actually type," use those two files instead.

## Background: SSRAE embedding layout

Verified in `core/tools/SSRAE/extractor.py`:
`emb = hstack[β_R, β_G, β_B, β_S_RG, β_S_GB, β_S_BR]` — six blocks of equal
shape `(9, Q+1)`. **This is NOT a contiguous-halves layout** — see the Layout
caveat below before implementing `embedding_variant` slicing.

### Layout caveat (verified 2026-08-23)

Each `beta` block has shape `(9, Q+1)` (9 = 3×3 patch dims). The extractor's
`torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1,-1))`
concatenates the six blocks along columns (`torch.hstack` on 2-D tensors is
axis=1), then flattens row-major. The result is **row-interleaved**: the six
blocks are NOT laid out as six contiguous chunks in the final length-`54*(Q+1)`
vector. So `emb[:len(emb)//2]` is not "spatial" and `emb[len(emb)//2:]` is not
"spectral". Correct slicing:

```python
M = emb.reshape(9, 6*(Q+1))          # rows = patch dims, column groups = [R,G,B,RG,GB,BR]
emb_spatial  = M[:, :3*(Q+1)].reshape(-1)   # Θ_R, Θ_G, Θ_B
emb_spectral = M[:, 3*(Q+1):].reshape(-1)   # Ω_RG, Ω_GB, Ω_BR
emb_full     = emb
```

(Equivalently: column-group `j` occupies columns `[j*(Q+1), (j+1)*(Q+1))` of
`M`.) The advisor's original intent — spatial vs. spectral halves — still
holds; only the naive `emb[:len(emb)//2]` implementation was wrong. See
`tests/test_ssrae_embedding_layout.py` and `.specs/quality/known-issues.md`.
**Refactor requirement**: the Phase 2 `EmbeddingProvider` must return
block-contiguous embeddings, or expose explicit block/column-group indices,
so naive halving cannot silently be wrong again.

Current `Q` value: **`Q = 13`**, hardcoded in `utils/data.py`
(`create_feature_maps_ssrae`, around line 139) as
`extractor = ColorFeatureExtractor(Q=Q)` with `Q = 13`. (VCTex uses a different
`Q = [5, 17]` at `utils/data.py` around line 70 — not the same hyperparameter,
do not conflate the two when writing the representation ablation.) Use `Q=13`
for the representation ablation below unless the advisor specifies otherwise.

## 6.1 Representation ablation

Fix `Q=13`. Three variants, all derived from the **same** cached full SSRAE
embedding (never recompute SSRAE three times):

| Variant | Definition | Status |
|---|---|---|
| `emb_full` | `emb` (all six column-groups) | Implemented — `dalmax/embeddings/variants.py::slice_embedding(v, "full", q)` (identity); this is what `SSRAEKmeansSampling`/`SSRAEKmeansHCSampling`/`RepresentationStrategy` use by default |
| `emb_spatial` | `emb.reshape(9, 6*(Q+1))[:, :3*(Q+1)].reshape(-1)` (R, G, B column-groups only — NOT `emb[:len(emb)//2]`, see Layout caveat above) | Implemented — `dalmax/embeddings/variants.py::slice_embedding(v, "spatial", q)`, set via the params JSON's `embedding.variant` key or the `--embedding_variant` CLI flag |
| `emb_spectral` | `emb.reshape(9, 6*(Q+1))[:, 3*(Q+1):].reshape(-1)` (RG, GB, BR column-groups only — NOT `emb[len(emb)//2:]`, see Layout caveat above) | Implemented — `dalmax/embeddings/variants.py::slice_embedding(v, "spectral", q)` |

**Implementation requirement — done (2026-08-23, Phase 2)**: `embedding.variant`
(`full | spatial | spectral`, params JSON, or `--embedding_variant` CLI override) **slices the
cached full embedding** — never recomputes SSRAE per variant. The cache is keyed on
`(dataset, extractor, Q, variant, split, pool_hash)`, but only `"full"` is ever written
(`dalmax/embeddings/cache.py`) — `spatial`/`spectral` are always derived from the loaded `"full"`
entry, so they cannot collide with each other or read a stale variant. Use the generic
`RepresentationStrategy` `--strategy_name`, not the `SSRAEKmeansHCSampling` preset, to reach the
`embedding.variant` config — the preset ignores the `"embedding"` block entirely (see
`.specs/experiments/ablation-study.md` §6.1's exact config for the full params JSON/CLI).

## 6.2 Hierarchy ablation

Fix `n_query = 100`, SSRAE full embeddings (`emb_full`). Vary `config_kmh`
(the hierarchy config in the params JSON — see `params_df_gpu_0.json`'s
`config_kmh: {n_clusters, n_levels, sample_sizes}`):

| L | k (n_clusters) | Note |
|---|---|---|
| 1 | `[50]` | |
| 2 | `[300, 100]` (alternative: `[100, 50]` — record both, run per advisor's choice) | |
| 3 | `[300, 100, 50]` | Close to current `params_df_gpu_*.json` shape (`[600,200,100]`, 3 levels) but not identical — this ablation cell uses the exact `[300,100,50]` triple |
| 4 | `[300, 100, 50, 25]` | Matches the `config_kmh` example seen in `core/query_strategies/ssl_ssrae_sampling.py`'s inline docstring comment |

**Implementation requirement — done (2026-08-23, Phase 2)**: hierarchy depth/cluster counts come
**entirely from config** — no hardcoded `'DANINHAS'` key lookup.
`dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection` takes `hierarchy` via
constructor injection (resolved per the actual `dataset_name`, not a literal string). The old
`core/query_strategies/ssl_ssrae_sampling.py:66` hardcode is unchanged but unreachable from
`demo.py`/`dalmax.cli` (dead code, see `.specs/architecture/current-state.md` §0). `sample_sizes`
semantics are now verified and documented (not a TBD any more): `sample_sizes[level]` only affects
*centroid-refinement resampling quality* at that level; the number of ids returned is controlled
entirely by `n_query` (independent of `sample_sizes`). Exact `sample_sizes` values for every `L`
row are in `.specs/experiments/ablation-study.md` §6.2's run table (derived proportionally to
`n_clusters`, flagged TBD for advisor sign-off but safe to run as-is).

## 6.3 Contribution of the two RNHAL stages

Three conditions, all macro F1 on `daninhas_full`:

1. **RNHAL (full)** — F1 taken from the already-executed reference experiments
   (`SSRAEKmeansHCSampling`, the current `run_pipe_gpu_*.sh` sweeps). No new
   run needed; pull the number from existing `results/` via
   `.claude/skills/results-reporting/SKILL.md`. Their `results.json` predates the Phase 2 macro-F1
   addition (weighted F1 only) — recompute macro F1 offline from `predictions.csv` for these
   specific runs, do not re-run them.
2. **Without representation module** — keep hierarchical selection, replace
   SSRAE embeddings with **ImageNet-pretrained ResNet embeddings** (penultimate
   layer of the existing ResNet50, `core/daninhas_model.py`). **Implemented (Phase 2)**:
   `dalmax/embeddings/resnet_imagenet_provider.py::ResNetImageNetProvider` — a provider swap fed
   into `RepresentationStrategy`, no new strategy class needed.
3. **Without hierarchical module** — SSRAE embeddings + **flat k-means**,
   selecting **random images from each cluster proportionally to the budget**.
   **This is a different strategy from the legacy `SSRAEKmeansSampling`**,
   which uses `k=n` (one cluster per requested sample) and picks the
   closest-to-centroid image per cluster. **Implemented (Phase 2)**:
   `dalmax/selection/flat_kmeans_proportional.py::FlatKMeansProportionalRandom` — a genuinely new
   selection strategy (cluster count independent of the query budget, random-not-closest picks,
   proportional per-cluster quotas), seeded from the experiment seed, not a config-driven variant
   of the old buggy class.

## Code capabilities — all implemented (2026-08-23, Phase 2)

- Embedding provider abstraction: SSRAE | VCTex | ResNet-ImageNet, one
  interface, swappable via config. — `dalmax/embeddings/{base,ssrae_provider,vctex_provider,
  resnet_imagenet_provider}.py`.
- `embedding_variant` slicing (`full | spatial | spectral`) on top of the
  provider's cached output. — `dalmax/embeddings/variants.py`.
- Configurable hierarchy (depth `L`, `n_clusters` per level, `sample_sizes` per
  level) with no hardcoded dataset key. — `dalmax/selection/hierarchical_kmeans.py`,
  `dalmax/config/schema.py::HierarchyConfig`.
- A proportional-cluster-sampling strategy (flat k-means + proportional random
  selection per cluster) as a distinct, registered strategy from
  `SSRAEKmeansSampling`. — `dalmax/selection/flat_kmeans_proportional.py`.

Every ablation cell is now a params-JSON/CLI-only change — see
`.specs/experiments/ablation-study.md` for the exact configs and
`.specs/use-cases/run-ablation.md` for the run order. What remains is executing these on the lab
machine and recording the resulting macro-F1 numbers.

## Running and reporting

Use `.claude/skills/running-experiments/SKILL.md` (lab machine, one
`config_kmh`/`embedding_variant` combination per run) and
`.claude/skills/results-reporting/SKILL.md` to turn the resulting `results/`
trees into the macro-F1 tables that feed the paper's ablation subsection. Track
progress with `.claude/commands/ablation-status.md`.
