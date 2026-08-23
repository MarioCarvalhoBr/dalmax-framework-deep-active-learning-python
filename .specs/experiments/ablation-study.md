# Ablation study specification

Status: **specification only — not implemented in code yet.** This is the
advisor-requested ablation section for the paper's
`\subsection{Ablation study}`. It is copied faithfully from
`prompt-master.md` §6, with added execution/run tables, required code
capabilities, expected artifacts, and the mapping to the paper's
`\subsubsection`s. A future implementation session should be able to work
from this file without re-deriving anything from the LaTeX or the codebase.

All three sub-studies are evaluated with **macro F1** (per the advisor's
request) on the held-out `test/` split of `daninhas_full`, using seeds and
budgets consistent with `experimental-protocol.md`.

> **Metrics discrepancy — read before implementing.** The codebase's actual
> metric function, `Data.calc_metrics_sklearn` in `utils/data.py`
> (line ~281–295 depending on future edits), computes precision/recall/F1
> with **`average='weighted'`**, not `average='macro'`. Every existing
> `results.json` in `results/dalmax1/`, `results/dalmax2/`, etc. was produced
> with weighted F1. To report **macro F1** for the ablation as specified
> here, either (a) add a macro-F1 computation alongside the existing
> weighted one (non-breaking, preserves comparability with prior results),
> or (b) recompute macro F1 offline from the per-image `predictions.csv`
> that `demo.py` already saves (`Real Class`, `Predicted Class` columns) —
> this requires no new experiment runs for already-executed configurations.
> Option (b) is the safer default so existing RNHAL-full reference numbers
> (§6.3) do not need to be re-run. See `research-rules/metrics.md`.

## 6.1 Representation ablation

- Fix **Q** at the value used in the SSRAE reference article / current
  experiments. **Verified current value: `Q = 13`**, hardcoded in
  `utils/data.py:139` (`create_feature_maps_ssrae`, comment: "The number of
  hidden neurons"). **TBD**: the advisor has separately asked to "fix Q per
  the SSRAE article" (`phd_files/artigo-original-tecnica-ssrae-manuscript.pdf`)
  — confirm whether the article's recommended Q matches 13 or whether the
  ablation (and the main RNHAL results) should be re-run at a different Q.
  Do not change `Q` silently; if it changes, the SSRAE embedding cache
  (`results/features_dict_ssrae.pkl`) becomes stale (see
  `research-rules/reproducibility.md`) and every downstream RNHAL result
  depends on it.
- SSRAE embedding layout (verified in `core/tools/SSRAE/extractor.py:99`):
  `emb = hstack[β_R, β_G, β_B, β_S_RG, β_S_GB, β_S_BR]` — code variable names are
  `beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B` in that exact
  order; `beta_S_R` is the R→G spectral fit, `beta_S_G` is G→B, `beta_S_B`
  is B→R. All six blocks are equal shape `(9, Q+1)` — **but see the Layout
  caveat below: the flattened vector is row-interleaved, not six contiguous
  blocks.**

### Layout caveat (verified 2026-08-23)

Each `rnn.beta` has shape `(9, Q+1)` (9 = 3×3 patch dims). The extractor does
`torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1,-1))`.
`torch.hstack` on 2-D tensors concatenates along axis=1 (columns), producing a
`(9, 6*(Q+1))` matrix; the subsequent row-major `.reshape((1, -1))` **interleaves
the six blocks per row** instead of laying them out as six contiguous chunks in
the final length-`54*(Q+1)` vector (756 for `Q=13`). Consequently:

- `emb[:len(emb)//2]` is **NOT** "spatial-only (R,G,B)".
- `emb[len(emb)//2:]` is **NOT** "spectral-only (RG,GB,BR)".

The correct slicing reshapes the flat vector back to `(9, 6*(Q+1))` and takes
column-groups (each group is `Q+1` columns wide):

```python
M = emb.reshape(9, 6*(Q+1))          # rows = patch dims, column groups = [R,G,B,RG,GB,BR]
emb_spatial  = M[:, :3*(Q+1)].reshape(-1)   # Θ_R, Θ_G, Θ_B
emb_spectral = M[:, 3*(Q+1):].reshape(-1)   # Ω_RG, Ω_GB, Ω_BR
emb_full     = emb
```

Equivalently: column-group `j` occupies columns `[j*(Q+1), (j+1)*(Q+1))` of `M`.
This was the advisor's original intent (spatial vs. spectral halves); the naive
`emb[:len(emb)//2]` implementation is simply wrong given the actual row-major
flatten. See `tests/test_ssrae_embedding_layout.py` for a test that pins this
down empirically, and `.specs/quality/known-issues.md` for the tracked issue.
**Refactor requirement**: the Phase 2 `EmbeddingProvider` (see
`.specs/architecture/target-architecture.md` §5) must return block-contiguous
embeddings, or expose explicit block indices/column-group boundaries, so that
naive halving is never silently wrong again.

- Variants to run:
  - `emb_spatial`  (as computed above) → spatial-only (R, G, B column-groups) → F1
  - `emb_spectral` (as computed above) → spectral-only (RG, GB, BR column-groups) → F1
  - `emb_full     = emb`                    → full (already implemented) → F1
- Implementation requirement: an `embedding_variant` config option
  (`full | spatial | spectral`) that **slices the cached full embedding**
  using the column-group logic above — never recompute SSRAE three times;
  the cache must be keyed so variants cannot collide (current cache at
  `results/features_dict_ssrae.pkl` has no key at all — see
  `research-rules/reproducibility.md`).
- Query strategy to use: `SSRAEKmeansHCSampling` (hierarchical) with the
  hierarchy config held fixed at whatever value is chosen as the ablation's
  reference hierarchy (see §6.2 — pick the winning config there first, or
  use the `run_pipe_gpu_0.sh` reference `n_clusters=[600,200,100]`,
  `n_levels=3`, `sample_sizes=[30,15,2]` as a provisional fixed point).

### 6.1 run table

| Variant       | embedding_variant | n_query | seeds  | strategy              | n_round | n_epoch |
|---------------|--------------------|---------|--------|-----------------------|---------|---------|
| Spatial-only  | `spatial`          | 100     | 1,2,3  | SSRAEKmeansHCSampling | 8       | 10      |
| Spectral-only | `spectral`         | 100     | 1,2,3  | SSRAEKmeansHCSampling | 8       | 10      |
| Full          | `full`             | 100     | 1,2,3  | SSRAEKmeansHCSampling | 8       | 10      |

TBD confirm with advisor whether §6.1 should also sweep `n_query ∈ {10,50,100}`
like the main protocol, or is scoped to a single representative budget
(n_query=100 chosen above as the largest/most informative budget — flag for
confirmation, not a settled decision).

## 6.2 Hierarchy ablation

All with `n_query = 100`, SSRAE full embeddings, varying the hierarchy
config (`config_kmh` in the params JSON):

| L (n_levels) | k (n_clusters)       | note |
|--------------|----------------------|------|
| 1            | [50]                 | |
| 2            | [300, 100]           | alternative: [100, 50] — record both, run per advisor's choice |
| 3            | [300, 100, 50]       | |
| 4            | [300, 100, 50, 25]   | matches the hardcoded fallback config in `ssl_ssrae_sampling.py:64` (dead docstring, but shows a prior value in use) |

Implementation requirement: hierarchy depth/cluster counts must come
**entirely from config** (no hardcoded `'DANINHAS'` key lookup — this is
currently broken: `core/query_strategies/ssl_ssrae_sampling.py:66` does
`config_kmh = self.params['DANINHAS']['config_kmh']` unconditionally,
regardless of `args.dataset_name`, so this strategy cannot currently run
against CIFAR10 even though it is a registered choice for both datasets —
see `known-issues.md`, owned by another batch). `sample_sizes` handling per
level must be specified explicitly: `sample_sizes[i]` is the number of
points sampled from level-`i` clusters during
`hierarchical_sampling.hierarchical_sampling(cl, target_size=n_query)`
(`core/tools/SSL/src/hierarchical_sampling.py` — TBD verify exact semantics
of how `sample_sizes` combines with `target_size` in that module; not
re-read in this batch, flag for the implementer).

**TBD**: no `sample_sizes` value is proposed above for the `L=1,2,3,4`
grid — the two params files on disk (`params_df_gpu_0.json`:
`n_clusters=[600,200,100]`/`sample_sizes=[30,15,2]`; `params_df_gpu_1.json`:
`n_clusters=[500,200,150]`/`sample_sizes=[60,30,2]`) show `sample_sizes`
scales with both `n_clusters` and `n_query`; the implementer must derive a
consistent `sample_sizes` per row of the table above (e.g. proportional to
`n_clusters[i]` and normalized so the hierarchy yields exactly `n_query=100`
samples) rather than inventing arbitrary values. Confirm the derivation
rule with the advisor before running.

### 6.2 run table

| Variant | config_kmh.n_levels | config_kmh.n_clusters | config_kmh.sample_sizes | n_query | seeds | strategy |
|---------|---------------------|------------------------|--------------------------|---------|-------|----------|
| L=1     | 1                   | [50]                   | TBD                      | 100     | 1,2,3 | SSRAEKmeansHCSampling |
| L=2 (a) | 2                   | [300, 100]             | TBD                      | 100     | 1,2,3 | SSRAEKmeansHCSampling |
| L=2 (b) | 2                   | [100, 50]              | TBD                      | 100     | 1,2,3 | SSRAEKmeansHCSampling |
| L=3     | 3                   | [300, 100, 50]         | TBD                      | 100     | 1,2,3 | SSRAEKmeansHCSampling |
| L=4     | 4                   | [300, 100, 50, 25]     | TBD                      | 100     | 1,2,3 | SSRAEKmeansHCSampling |

## 6.3 Contribution of the two RNHAL stages

- **RNHAL (full)**: F1 taken from the **already-executed** reference runs
  (`results/dalmax1/daninhas_full/.../SSRAEKmeansHCSampling/`,
  `results/dalmax2/daninhas_full/.../SSRAEKmeansHCSampling/`, n_query=100 —
  see `baseline-results.md`). No new runs needed for this row.
- **Without representation module**: keep hierarchical selection, replace
  SSRAE embeddings with **ImageNet-pretrained ResNet embeddings**
  (penultimate layer of the existing ResNet50 — new embedding provider, not
  present in the codebase today). Strategy: hierarchical selection
  (`SSLStrategy` machinery in `ssl_ssrae_sampling.py`) fed by the new
  provider instead of `create_feature_maps_ssrae`.
- **Without hierarchical module**: SSRAE embeddings + **flat k-means**,
  selecting **random images from each cluster proportionally to the
  budget**. Note this is **different** from the current
  `SSRAEKmeansSampling` (`core/query_strategies/ssrae_kmeans_sampling.py`),
  which:
  - sets `n_clusters = n` (i.e., `n_query`), not a smaller flat cluster count,
  - and picks the single sample **closest to the centroid** per cluster
    (`np.argmin(distances)`), not a random/proportional sample.

  The spec explicitly requires a **new** strategy — "proportional-random
  flat k-means" — with a genuinely separate cluster count from `n_query`
  and random-not-closest selection weighted by budget share.
  `SSRAEKmeansSampling` also hardcodes `KMeans(random_state=3)`
  (`ssrae_kmeans_sampling.py:23`), ignoring the experiment seed — do not
  reuse it as-is for this ablation row; either fix the seed handling in a
  new class or make it a config-driven variant of the existing one.

### 6.3 run table

| Variant                      | Representation           | Selection                                   | F1 source |
|-------------------------------|---------------------------|----------------------------------------------|-----------|
| RNHAL (full)                  | SSRAE (Φ)                 | Hierarchical k-means (Γ)                     | Reuse `results/dalmax{1,2}` reference runs, n_query=100 |
| w/o representation module      | ImageNet ResNet50 penult. | Hierarchical k-means (Γ)                     | New runs, all seeds, n_query per protocol |
| w/o hierarchical module        | SSRAE (Φ)                 | Flat k-means, proportional-random per cluster | New runs, all seeds, n_query per protocol |

## Code capabilities the refactor must provide

To make the three ablations one-config-line runs (Phase 2/3 of
`architecture/refactor-plan.md`, owned by another batch — cross-reference
once that file exists):

1. **Embedding provider abstraction** — a common interface for
   `SSRAE | VCTex | ResNet-ImageNet` embeddings, so §6.3's "without
   representation module" variant is a provider swap, not a new strategy
   class.
2. **`embedding_variant` slicing** (`full | spatial | spectral`) applied on
   top of any provider's cached full embedding — needed for §6.1.
3. **Configurable hierarchy** — `n_levels`/`n_clusters`/`sample_sizes` fully
   driven by the params JSON with no hardcoded dataset key, needed for §6.2
   and to unblock `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling` on
   CIFAR10.
4. **Proportional-cluster-sampling selection strategy** — flat k-means with
   a configurable cluster count independent of `n_query`, sampling randomly
   from each cluster in proportion to the query budget — needed for §6.3's
   "without hierarchical module" row.
5. **Cache keys** for every cached embedding
   (`dataset × extractor × Q × embedding_variant`), so running §6.1's three
   variants back-to-back cannot silently reuse a stale full-embedding
   cache computed for a different Q or dataset.

## Output artifacts

Per run: the same artifacts `demo.py` already produces
(`results.json`, `predictions.csv`, `confusion_matrix.pdf`, per-metric
plots, `log-dalmax.log`, saved model) under a results path that encodes the
ablation variant, e.g.
`results/ablation_representation/daninhas_full/SEED_{seed}/NQ_100_NIL_100_NR_8_NE_10/{embedding_variant}/`
and analogously for `results/ablation_hierarchy/.../{L}_{cluster_config_id}/`
and `results/ablation_stage_contribution/.../{variant_name}/` (exact naming
TBD — should extend, not replace, the naming convention in
`experimental-protocol.md`, adding one path segment for the ablation axis).
Aggregation: run `utils/report/2_report_build_chunk_results.py` and
`4_report_build_average_results.py` (see `use-cases/generate-report.md`)
per ablation family to get seed-averaged macro-F1 tables, then hand-author
the LaTeX table (or delegate to the `paper-liaison` agent, owned by another
batch).

## Mapping to the paper's `\subsubsection`s

| This spec section | Paper subsubsection (working title) |
|---|---|
| §6.1 Representation ablation | "Representation ablation" / effect of spatial vs. spectral SSRAE components |
| §6.2 Hierarchy ablation | "Hierarchy ablation" / effect of hierarchy depth `L` and cluster counts `k_i` |
| §6.3 Contribution of the two RNHAL stages | "Contribution of the two RNHAL stages" / ablating `Φ` and `Γ` independently |

TBD: confirm exact subsubsection titles once the advisor finalizes the
`\subsection{Ablation study}` outline in `phd_files/Active_Learning_Mario/`.
