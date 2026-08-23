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
| `emb_full` | `emb` (all six column-groups) | Already implemented — this is what `SSRAEKmeansSampling`/`SSRAEKmeansHCSampling` use today |
| `emb_spatial` | `emb.reshape(9, 6*(Q+1))[:, :3*(Q+1)].reshape(-1)` (R, G, B column-groups only — NOT `emb[:len(emb)//2]`, see Layout caveat above) | Not implemented — needs `embedding_variant` config option |
| `emb_spectral` | `emb.reshape(9, 6*(Q+1))[:, 3*(Q+1):].reshape(-1)` (RG, GB, BR column-groups only — NOT `emb[len(emb)//2:]`, see Layout caveat above) | Not implemented — needs `embedding_variant` config option |

**Implementation requirement**: an `embedding_variant` config option
(`full | spatial | spectral`) that **slices the cached full embedding** —
never recompute SSRAE per variant. The cache must be keyed so variants cannot
collide (today `results/features_dict_ssrae.pkl` has no such key — see
`.claude/rules/reproducibility.md`; this is the first thing to fix before
running this ablation, or the `spatial`/`spectral` run will silently read a
`full`-embedding cache file left over from a previous run).

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

**Implementation requirement**: hierarchy depth/cluster counts must come
**entirely from config** — no hardcoded `'DANINHAS'` key lookup. This is
currently broken: `core/query_strategies/ssl_ssrae_sampling.py` reads
`self.params['DANINHAS']['config_kmh']` literally, so this ablation cannot run
against any dataset name other than `DANINHAS` without a code fix first (see
`.specs/quality/known-issues.md`). `sample_sizes` handling per level must also
be specified explicitly per `L` (do not assume a formula — define the
`sample_sizes` list for each row above alongside its `n_clusters` when writing
the run config).

## 6.3 Contribution of the two RNHAL stages

Three conditions, all macro F1 on `daninhas_full`:

1. **RNHAL (full)** — F1 taken from the already-executed reference experiments
   (`SSRAEKmeansHCSampling`, the current `run_pipe_gpu_*.sh` sweeps). No new
   run needed; pull the number from existing `results/` via
   `.claude/skills/results-reporting/SKILL.md`.
2. **Without representation module** — keep hierarchical selection, replace
   SSRAE embeddings with **ImageNet-pretrained ResNet embeddings** (penultimate
   layer of the existing ResNet50, `core/daninhas_model.py`). This needs a
   **new embedding provider** — not implemented today.
3. **Without hierarchical module** — SSRAE embeddings + **flat k-means**,
   selecting **random images from each cluster proportionally to the budget**.
   **This is a different strategy from the current `SSRAEKmeansSampling`**,
   which uses `k=n` (one cluster per requested sample) and picks the
   closest-to-centroid image per cluster — the ablation's "no hierarchical
   module" condition needs a **new** strategy (flat k-means with a cluster
   count independent of the query budget, then proportional random sampling
   within each cluster), not a re-run of the existing one. Do not conflate the
   two when implementing or when writing up results.

## Code capabilities the refactor must provide (Phase 2/3, see
`.specs/architecture/refactor-plan.md`)

- Embedding provider abstraction: SSRAE | VCTex | ResNet-ImageNet, one
  interface, swappable via config.
- `embedding_variant` slicing (`full | spatial | spectral`) on top of the
  provider's cached output.
- Configurable hierarchy (depth `L`, `n_clusters` per level, `sample_sizes` per
  level) with no hardcoded dataset key.
- A proportional-cluster-sampling strategy (flat k-means + proportional random
  selection per cluster) as a distinct, registered strategy from
  `SSRAEKmeansSampling`.

Each of these should become a one-config-line change once Phase 2 of the
refactor plan is done — that is the acceptance bar for calling the ablations
"implemented."

## Running and reporting

Use `.claude/skills/running-experiments/SKILL.md` (lab machine, one
`config_kmh`/`embedding_variant` combination per run) and
`.claude/skills/results-reporting/SKILL.md` to turn the resulting `results/`
trees into the macro-F1 tables that feed the paper's ablation subsection. Track
progress with `.claude/commands/ablation-status.md`.
