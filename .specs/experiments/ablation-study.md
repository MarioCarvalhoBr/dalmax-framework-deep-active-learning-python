# Ablation study specification

Status: **materialized as of 2026-08-23 (Phase 3 landed) — configs, scripts, and report tooling
all exist on disk and are CPU-smoke-tested (11/11); not yet run on the lab machine.** See the
"Materialized files" section near the end of this document for the exact mapping from every run
table row below to its `files_config/ablations/*.json` file and GPU run script.

Previous status: **implementable via config as of 2026-08-23 (Phase 2 landed) — not yet run.** This is the
advisor-requested ablation section for the paper's `\subsection{Ablation study}`. It was originally
copied faithfully from `prompt-master.md` §6; the "Code capabilities" section's requirements are now
all implemented (`dalmax/embeddings/`, `dalmax/selection/`, `dalmax.query_strategies.
RepresentationStrategy` — see `.specs/architecture/refactor-plan.md` Phase 2's "ablation enablers
checklist", fully checked off). Every run table below now has exact params-JSON snippets and CLI
invocations using the new `"embedding"`/`"selection"` config blocks — no further code changes are
needed to execute any of the three sub-studies; what remains is running them on the lab machine
(`.specs/infrastructure/execution-environments.md`) and recording the resulting macro-F1 numbers
here and in `baseline-results.md`. A future implementation session should be able to work from this
file without re-deriving anything from the LaTeX or the codebase.

All three sub-studies are evaluated with **macro F1** (per the advisor's
request) on the held-out `test/` split of `daninhas_full`, using seeds and
budgets consistent with `experimental-protocol.md`.

> **Metrics discrepancy — resolved 2026-08-23 (Phase 2).** Option (a) below was implemented:
> `dalmax/data/datasets.py::Data.calc_metrics` (was `utils/data.py`, moved in Phase 4; new method) computes **both** weighted and macro
> precision/recall/F1 in one pass, and `dalmax/experiment/reporter.py` writes all of them into
> `results.json` (`all_precision_macro`/`all_recall_macro`/`all_f1_macro`, additive — the legacy
> weighted keys are unchanged). Any run through `dalmax.cli`/`demo.py` from this commit onward
> already has macro F1 in its `results.json`; no offline recomputation from `predictions.csv` is
> needed for **new** runs. It is **still needed for already-executed pre-Phase-2 reference runs**
> (`results/dalmax1/`, `results/dalmax2/`, used as-is by §6.3's "RNHAL (full)" row) — those
> `results.json` files predate this fix and only have weighted F1; recompute macro F1 for them
> offline from their `predictions.csv` (`Real Class`, `Predicted Class` columns), do not re-run
> them. See `research-rules/metrics.md` for the exact `calc_metrics` field names to use.
>
> Original text, kept for context: the codebase's metric function,
> `Data.calc_metrics_sklearn` (was in `utils/data.py:281-295`, now `dalmax/data/datasets.py` since
> Phase 4's move), computes precision/recall/F1 with
> `average='weighted'`, not `average='macro'` — every existing `results.json` in `results/dalmax1/`,
> `results/dalmax2/`, etc. was produced with weighted F1 only, since `calc_metrics_sklearn` itself
> was left unmodified (only a new sibling method, `calc_metrics`, was added).

## 6.1 Representation ablation

- Fix **Q** at the value used in the SSRAE reference article / current
  experiments. **Verified value: `Q = 13`**, originally hardcoded in
  `utils/data.py:139` (`create_feature_maps_ssrae`, comment: "The number of
  hidden neurons"; that file was deleted in Phase 4). `Q` is now config-driven
  (`EmbeddingConfig.q`, `dalmax/config/loader.py`), still defaulting to `13` for
  SSRAE. **TBD**: the advisor has separately asked to "fix Q per
  the SSRAE article" (`phd_files/artigo-original-tecnica-ssrae-manuscript.pdf`)
  — confirm whether the article's recommended Q matches 13 or whether the
  ablation (and the main RNHAL results) should be re-run at a different Q.
  Do not change `Q` silently; if it changes, the SSRAE embedding cache
  (`results/features_dict_ssrae.pkl`) becomes stale (see
  `research-rules/reproducibility.md`) and every downstream RNHAL result
  depends on it.
- SSRAE embedding layout (verified in `dalmax/tools/SSRAE/extractor.py`, was
  `core/tools/SSRAE/extractor.py:99` before Phase 4's move):
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
  the cache must be keyed so variants cannot collide.
  **Implemented (Phase 2)**: `dalmax/embeddings/variants.py::slice_embedding`
  is exactly this function; `dalmax/embeddings/cache.py::EmbeddingCache`
  always stores/loads the `"full"` variant only (see `.claude/rules/reproducibility.md`
  "Embedding cache discipline") — running `spatial`/`spectral`/`full` back-to-back
  against the same `(dataset, Q, pool)` extracts SSRAE features exactly once.
- Query strategy to use: `RepresentationStrategy` (the generic Phase 2 name,
  not the `SSRAEKmeansHCSampling` preset — the preset **pins** `extractor="ssrae"`
  and **ignores** the params JSON's `"embedding"` block entirely, including
  `variant`; see `dalmax/query_strategies/registry.py::REPRESENTATION_PRESET_Q`'s
  docstring). Use `"embedding": {"extractor": "ssrae", "q": 13, "variant": "spatial"|"spectral"|"full"}`
  with `"selection": {"method": "hierarchical", "hierarchy": {...}}` held fixed
  at whatever value is chosen as the ablation's reference hierarchy (see §6.2
  — pick the winning config there first, or use the `run_pipe_gpu_0.sh`
  reference `n_clusters=[600,200,100]`, `n_levels=3`, `sample_sizes=[30,15,2]`
  as a provisional fixed point, as done in the exact config below).

### 6.1 run table

| Variant       | embedding_variant | n_query | seeds  | strategy_name          | n_round | n_epoch |
|---------------|--------------------|---------|--------|------------------------|---------|---------|
| Spatial-only  | `spatial`          | 100     | 1,2,3  | RepresentationStrategy | 8       | 10      |
| Spectral-only | `spectral`         | 100     | 1,2,3  | RepresentationStrategy | 8       | 10      |
| Full          | `full`             | 100     | 1,2,3  | RepresentationStrategy | 8       | 10      |

TBD confirm with advisor whether §6.1 should also sweep `n_query ∈ {10,50,100}`
like the main protocol, or is scoped to a single representative budget
(n_query=100 chosen above as the largest/most informative budget — flag for
confirmation, not a settled decision).

### 6.1 exact params JSON + CLI (Phase 2 syntax)

One params JSON per variant, differing only in `embedding.variant` (`spatial` shown; swap for
`spectral`/`full`):

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10,
        "n_drop": 10,
        "n_classes": 5,
        "train_args": {"batch_size": 64, "num_workers": 4},
        "test_args": {"batch_size": 64, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "ssrae", "q": 13, "variant": "spatial"},
        "selection": {
            "method": "hierarchical",
            "hierarchy": {"n_clusters": [600, 200, 100], "n_levels": 3, "sample_sizes": [30, 15, 2]}
        }
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
    --params_json params_ablation_6_1_spatial.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablation_representation/ --device cuda
```

**Note on `n_classes`**: `5`, matching `params_df_gpu_0.json` and the five class
subdirectories under `DATA/daninhas_full/` (verified 2026-08-23; an earlier audit snippet
mistakenly carried `6`).

## 6.2 Hierarchy ablation

All with `n_query = 100`, SSRAE full embeddings, varying the hierarchy
config (`config_kmh` in the params JSON):

| L (n_levels) | k (n_clusters)       | note |
|--------------|----------------------|------|
| 1            | [50]                 | |
| 2            | [300, 100]           | alternative: [100, 50] — record both, run per advisor's choice |
| 3            | [300, 100, 50]       | |
| 4            | [300, 100, 50, 25]   | matches the hardcoded fallback config in `ssl_ssrae_sampling.py:64` (was in `core/query_strategies/`, deleted in Phase 4; dead docstring, but shows a prior value in use) |

Implementation requirement: hierarchy depth/cluster counts must come
**entirely from config** (no hardcoded `'DANINHAS'` key lookup).
**Resolved (Phase 2)**: `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`
takes `hierarchy` (n_clusters/n_levels/sample_sizes) via its constructor, injected from
`config.dataset.selection.hierarchy` — no hardcoded dataset-name key anywhere in this path. The
legacy `core/query_strategies/ssl_ssrae_sampling.py:66` hardcode existed only in a file that is now
**deleted (Phase 4)** — there is no legacy path left for it to block; the `SSRAEKmeansHCSampling`
CLI name (preset or generic `RepresentationStrategy`) always resolves through the new config path
today, with `extractor`/`q` pinned to the legacy SSRAE `Q=13` value for the preset — either name is
safe for this ablation.

**`sample_sizes` semantics — resolved (2026-08-23, Phase 2), replaces the earlier TBD.** Verified by
reading `dalmax/tools/SSL/src/hierarchical_kmeans_gpu.py`/`hierarchical_sampling.py` (was
`core/tools/SSL/src/...` before Phase 4's move) while
implementing `dalmax/selection/hierarchical_kmeans.py` (full derivation in that module's
docstring):

- `sample_sizes[level]` and the final selection budget (`n_query`) are **independent knobs**, not
  composed the way the original TBD guessed. `sample_sizes[level]` only affects the
  **centroid-refinement resampling** performed *inside* `hierarchical_kmeans_with_resampling` for
  that level: at each of `n_resamples` iterations, up to `sample_sizes[level]` points are drawn
  (closest-to-centroid, by default) from every cluster at that level and used to recompute that
  level's centroids. It is a clustering-quality/runtime knob — larger values use more points per
  resampling step (closer to the full level population, more expensive), smaller values are
  cheaper and noisier. If `sample_sizes[level] <= 1`, no resampling happens for that level.
- The **number of ids finally returned is controlled entirely by `n_query`** (passed as
  `target_size` to `hierarchical_sampling.hierarchical_sampling`), which recursively splits
  `n_query` across the top-level clusters (and their subclusters) in proportion to cluster size
  (largest-remainder-style) — **`sample_sizes` is never consulted for this**. So the earlier
  guess that `sample_sizes` "combines with `target_size`" to determine the output count was wrong;
  `n_query` alone determines it.
- **Derivation rule used for the table below** (consistent with the two params files' `Q≈0.05`
  ratio, `sample_sizes[i] ≈ round(n_clusters[i] * 0.05)`, floor 2): this is a reasonable,
  documented convention, not an advisor-confirmed one — **TBD confirm with the advisor** before
  treating these as final, though changing them only affects clustering quality, not `n_query`'s
  output count, so re-running with a different `sample_sizes` choice does not change what "L, k"
  means for this ablation's headline comparison.

### 6.2 run table

| Variant | selection.hierarchy.n_levels | selection.hierarchy.n_clusters | selection.hierarchy.sample_sizes | n_query | seeds | strategy_name |
|---------|------------------------------|----------------------------------|-------------------------------------|---------|-------|----------------|
| L=1     | 1                            | [50]                              | [3]                                  | 100     | 1,2,3 | RepresentationStrategy |
| L=2 (a) | 2                            | [300, 100]                        | [15, 5]                              | 100     | 1,2,3 | RepresentationStrategy |
| L=2 (b) | 2                            | [100, 50]                         | [5, 3]                               | 100     | 1,2,3 | RepresentationStrategy |
| L=3     | 3                            | [300, 100, 50]                    | [15, 5, 3]                           | 100     | 1,2,3 | RepresentationStrategy |
| L=4     | 4                            | [300, 100, 50, 25]                | [15, 5, 3, 2]                        | 100     | 1,2,3 | RepresentationStrategy |

### 6.2 exact params JSON + CLI (Phase 2 syntax, L=3 shown)

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10,
        "n_drop": 10,
        "n_classes": 5,
        "train_args": {"batch_size": 64, "num_workers": 4},
        "test_args": {"batch_size": 64, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "ssrae", "q": 13, "variant": "full"},
        "selection": {
            "method": "hierarchical",
            "hierarchy": {"n_clusters": [300, 100, 50], "n_levels": 3, "sample_sizes": [15, 5, 3]}
        }
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
    --params_json params_ablation_6_2_L3.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablation_hierarchy/ --device cuda
```

Swap `embedding.variant`/`selection.hierarchy` per row for the other four configs (L=1, L=2a, L=2b,
L=4) — one params JSON per row, same CLI shape. (verify against the live dataset before running).

## 6.3 Contribution of the two RNHAL stages

- **RNHAL (full)**: F1 taken from the **already-executed** reference runs
  (`results/dalmax1/daninhas_full/.../SSRAEKmeansHCSampling/`,
  `results/dalmax2/daninhas_full/.../SSRAEKmeansHCSampling/`, n_query=100 —
  see `baseline-results.md`). No new runs needed for this row.
- **Without representation module**: keep hierarchical selection, replace
  SSRAE embeddings with **ImageNet-pretrained ResNet embeddings**
  (penultimate layer of the existing ResNet50). **Implemented (Phase 2)**:
  `dalmax/embeddings/resnet_imagenet_provider.py::ResNetImageNetProvider` —
  torchvision `resnet50(ResNet50_Weights.IMAGENET1K_V1)`, classification head
  replaced with `nn.Identity()`, returns the pooled 2048-d penultimate vector
  per image, `eval()` mode, no gradient. Fed to
  `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`
  through the generic `RepresentationStrategy` (no new strategy class needed
  — this row is a provider swap, exactly as ADR 0003 intended).
- **Without hierarchical module**: SSRAE embeddings + **flat k-means**,
  selecting **random images from each cluster proportionally to the
  budget**. **Implemented (Phase 2)**:
  `dalmax/selection/flat_kmeans_proportional.py::FlatKMeansProportionalRandom` —
  a genuinely new selection strategy, deliberately distinct from
  `FlatKMeansClosest`/the legacy `SSRAEKmeansSampling`:
  - cluster count `k = n_clusters` (constructor arg, independent of the
    query budget) rather than `k = n_query`;
  - per-cluster quota is `n_query` distributed proportionally to relative
    cluster size (largest-remainder/Hamilton rounding, so quotas sum exactly
    to `n_query`);
  - picks are **random** within each cluster (`rng.choice(..., replace=False)`),
    not closest-to-centroid;
  - the k-means seed is derived from `rng` (via
    `dalmax.seeding.derive_seed(config.seed, "selection")`), never a hardcoded
    literal — this row's implementation does not inherit `SSRAEKmeansSampling`'s
    `KMeans(random_state=3)` bug (KI-5) at all, since it is new code, not a
    config-driven variant of the old class.

### 6.3 run table

| Variant                      | Representation           | Selection                                   | strategy_name / config | F1 source |
|-------------------------------|---------------------------|----------------------------------------------|--------------------------|-----------|
| RNHAL (full)                  | SSRAE (Φ)                 | Hierarchical k-means (Γ)                     | `SSRAEKmeansHCSampling` preset (unchanged behavior) | Reuse `results/dalmax{1,2}` reference runs, n_query=100 — recompute macro F1 offline from their `predictions.csv` (pre-Phase-2 `results.json` has weighted F1 only, see the "Metrics discrepancy" note above) |
| w/o representation module      | ImageNet ResNet50 penult. | Hierarchical k-means (Γ)                     | `RepresentationStrategy`, `embedding.extractor="resnet_imagenet"`, `selection.method="hierarchical"` | New runs, all seeds, n_query per protocol |
| w/o hierarchical module        | SSRAE (Φ)                 | Flat k-means, proportional-random per cluster | `RepresentationStrategy`, `embedding.extractor="ssrae"`, `selection.method="flat_proportional"` | New runs, all seeds, n_query per protocol |

### 6.3 exact params JSON + CLI (Phase 2 syntax)

**Row 2 — without representation module** (ResNet-ImageNet + hierarchical, same reference hierarchy
as §6.1/§6.2's provisional fixed point):

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10, "n_drop": 10, "n_classes": 5,
        "train_args": {"batch_size": 64, "num_workers": 4},
        "test_args": {"batch_size": 64, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "resnet_imagenet", "q": null, "variant": "full"},
        "selection": {
            "method": "hierarchical",
            "hierarchy": {"n_clusters": [600, 200, 100], "n_levels": 3, "sample_sizes": [30, 15, 2]}
        }
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
    --params_json params_ablation_6_3_resnet_hier.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablation_stage_contribution/ --device cuda
```

**Row 3 — without hierarchical module** (SSRAE full + flat proportional-random):

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10, "n_drop": 10, "n_classes": 5,
        "train_args": {"batch_size": 64, "num_workers": 4},
        "test_args": {"batch_size": 64, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "ssrae", "q": 13, "variant": "full"},
        "selection": {"method": "flat_proportional"}
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> python demo.py \
    --params_json params_ablation_6_3_ssrae_flat.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablation_stage_contribution/ --device cuda
```

Note `"selection": {"method": "flat_proportional"}` has no `"hierarchy"` key — `SelectionConfig`
rejects a `hierarchy` block for any non-`"hierarchical"` method (`dalmax/config/schema.py`), so
omitting it entirely is correct, not an oversight. `FlatKMeansProportionalRandom`'s cluster count
(`n_clusters` constructor arg) defaults to `n_query` when unset — as of Phase 2, `build_strategy`
(`dalmax/query_strategies/registry.py`) does not thread a separate `selection.n_clusters` params-JSON
key to it, so this row currently runs with `k = n_query` rather than a cluster count independently
chosen from the budget. **TBD**: confirm with the advisor whether this satisfies the ablation's
"genuinely separate cluster count from `n_query`" intent (see the "Without hierarchical module"
bullet above), or whether a `selection.n_clusters` config key should be added to
`dalmax/config/schema.py::SelectionConfig`/`loader.py` before this row is run on the lab machine —
flag before running, this is a small, well-scoped addition if needed. 
## Code capabilities the refactor must provide

**Status: all five capabilities below are implemented as of Phase 2 (2026-08-23)** — see
`.specs/architecture/refactor-plan.md`'s "ablation enablers checklist" for the itemized evidence.
Kept here as the original requirements list for traceability.

1. **Embedding provider abstraction** — a common interface for
   `SSRAE | VCTex | ResNet-ImageNet` embeddings, so §6.3's "without
   representation module" variant is a provider swap, not a new strategy
   class. — **Done**: `dalmax/embeddings/{base,ssrae_provider,vctex_provider,resnet_imagenet_provider}.py`.
2. **`embedding_variant` slicing** (`full | spatial | spectral`) applied on
   top of any provider's cached full embedding — needed for §6.1. — **Done**:
   `dalmax/embeddings/variants.py::slice_embedding`.
3. **Configurable hierarchy** — `n_levels`/`n_clusters`/`sample_sizes` fully
   driven by the params JSON with no hardcoded dataset key, needed for §6.2
   and to unblock `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling` on
   CIFAR10. — **Done**: `dalmax/selection/hierarchical_kmeans.py`,
   `dalmax/config/schema.py::HierarchyConfig`/`SelectionConfig`.
4. **Proportional-cluster-sampling selection strategy** — flat k-means with
   a configurable cluster count independent of `n_query`, sampling randomly
   from each cluster in proportion to the query budget — needed for §6.3's
   "without hierarchical module" row. — **Done**:
   `dalmax/selection/flat_kmeans_proportional.py::FlatKMeansProportionalRandom`
   (see §6.3's exact-config note above for the one remaining TBD: whether its
   `n_clusters` needs its own params-JSON key rather than defaulting to `n_query`).
5. **Cache keys** for every cached embedding
   (`dataset × extractor × Q × embedding_variant`), so running §6.1's three
   variants back-to-back cannot silently reuse a stale full-embedding
   cache computed for a different Q or dataset. — **Done, plus one more key
   component**: `dalmax/embeddings/cache.py::EmbeddingCache`, keyed on
   `(dataset, extractor, Q, variant, split, pool_hash)` — `pool_hash` was
   added beyond the original spec because the embedded pool also depends on
   `--seed`/`--n_init_labeled`.

## Output artifacts

**Resolved (2026-08-23, Phase 3) — replaces the earlier TBD naming.** Per run: the same artifacts
`demo.py` already produces (`results.json`, `predictions.csv`, `confusion_matrix.pdf`, per-metric
plots, `log-dalmax.log`, saved model, `run_metadata.json`) under
`results/ablations/{study}/{config}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/RepresentationStrategy/`
— `{study}` is `6_1`/`6_2`/`6_3`, `{config}` is the `files_config/ablations/{config}.json` basename
(e.g. `rep_spatial`, `hier_L3`, `stage_no_hierarchy`), and everything from `{dataset_folder}`
onward is the unmodified `experimental-protocol.md` naming convention (`dalmax/experiment/runner.py::
results_dir_for`) — the ablation axis is encoded as two extra leading path segments
(`{study}/{config}`), not by changing the existing convention. `scripts/ablations/run_ablation_gpu_0.sh`
/ `run_ablation_gpu_1.sh` pass `--dir_results results/ablations/{study}/{config}/` per config, and
`scripts/ablations/smoke_ablations.sh` mirrors the same `{study}/{config}` nesting under
`results/smoke_ablations/` for CPU smoke runs. Aggregation:
`python -m dalmax.reporting.ablation_report --root results/ablations --out paper_drafts/ablation_tables`
(see "Materialized files" below) walks exactly this layout and writes seed-averaged macro-F1 tables
directly — `dalmax/reporting/chunk_results.py`/`average_results.py` (was
`utils/report/2_report_build_chunk_results.py`/`4_report_build_average_results.py` before Phase 4's
move/rename) remain the tool for the main `results/dalmax{1,2}/` sweeps
(`use-cases/generate-report.md`), not for this ablation-specific tree.

## Materialized files (Phase 3, 2026-08-23)

Every run-table row above is now a real file, not just a JSON snippet in this document. Full-scale
configs live in `files_config/ablations/`; `files_config/ablations/micro/` mirrors each one for
local CPU smoke-testing (`make smoke-ablations`, 11/11 passing against `DATA/daninhas_micro/`) —
see that folder's own `README.md` for every config's exact `embedding`/`selection` values and the
`sample_sizes`/micro-hierarchy derivation rules. `tests/test_ablation_configs.py` loads all 22
files (11 full-scale + 11 micro) through `dalmax.config.loader.load_experiment_config` and asserts
each resolves to the extractor/variant/method/hierarchy its filename implies.

| §  | Row | Params JSON | GPU script |
|----|-----|--------------|------------|
| 6.1 | Full | `files_config/ablations/rep_full.json` | `run_ablation_gpu_1.sh` |
| 6.1 | Spatial-only | `files_config/ablations/rep_spatial.json` | `run_ablation_gpu_0.sh` |
| 6.1 | Spectral-only | `files_config/ablations/rep_spectral.json` | `run_ablation_gpu_1.sh` |
| 6.2 | L=1, k=[50] | `files_config/ablations/hier_L1.json` | `run_ablation_gpu_0.sh` |
| 6.2 | L=2, k=[300,100] | `files_config/ablations/hier_L2a.json` | `run_ablation_gpu_1.sh` |
| 6.2 | L=2, k=[100,50] | `files_config/ablations/hier_L2b.json` | `run_ablation_gpu_0.sh` |
| 6.2 | L=3, k=[300,100,50] | `files_config/ablations/hier_L3.json` | `run_ablation_gpu_1.sh` |
| 6.2 | L=4, k=[300,100,50,25] | `files_config/ablations/hier_L4.json` | `run_ablation_gpu_1.sh` |
| 6.3 | RNHAL (full) | `files_config/ablations/stage_full.json` | `run_ablation_gpu_1.sh` |
| 6.3 | w/o representation module | `files_config/ablations/stage_no_representation.json` | `run_ablation_gpu_0.sh` |
| 6.3 | w/o hierarchical module | `files_config/ablations/stage_no_hierarchy.json` | `run_ablation_gpu_1.sh` |

(`scripts/ablations/run_ablation_gpu_0.sh` = 5 configs, `run_ablation_gpu_1.sh` = 6 configs, split
by expected relative cost — hierarchical selection over a large/multi-level hierarchy dominates
runtime, not training; see `run_ablation_gpu_0.sh`'s header comment for the full rationale.) Both
scripts sweep `SEEDS=(1 2 3)`, `n_query=100`, `n_round=8`, `n_init_labeled` left at the CLI default
(100), `--device cuda`, `--strategy_name RepresentationStrategy`, and log any failing
`(config, seed)` to `results/ablations/gpu{0,1}_failures.log` instead of aborting the batch.

Aggregation: `dalmax/reporting/ablation_report.py` (`python -m dalmax.reporting.ablation_report
--root results/ablations --out paper_drafts/ablation_tables`) walks
`{root}/{study}/{config}/{dataset}/SEED_*/NQ_*/RepresentationStrategy/results.json`, computes
per-config mean ± std across seeds of both the final-round value and the across-rounds mean (for
`all_f1_macro` and `all_f1_score`), and writes `ablation_summary.csv` plus one booktabs
`ablation_6_{1,2,3}.md`/`.tex` per sub-study (a config with zero discovered runs renders `TBD`).
Unit-tested against a synthetic tree in `tests/test_ablation_report.py`; also run once against real
(smoke) output — `results/smoke_ablations/` — to confirm the walk pattern matches actual
`demo.py`/`ExperimentRunner` output byte-for-byte, not just the synthetic fixture's assumptions.

**Ambiguities resolved while materializing these files** (none required a code change):

- **§6.1's reference hierarchy**: used the `run_pipe_gpu_0.sh`/`params_df_gpu_0.json` reference
  (`n_clusters=[600,200,100]`, `sample_sizes=[30,15,2]`) exactly as this document's own §6.1 exact-config
  note already specifies, rather than waiting on a §6.2 "winning config" (§6.2 itself hasn't been run
  yet, so there is no winner to pick).
- **`train_args`/`test_args` batch size**: this document's exact-config JSON snippets show
  `batch_size: 64`, but the materialized files use `256` (matching `params_df_gpu_0.json` and the
  refactor-plan Phase 3 task's explicit instruction to mirror that file) — a deliberate deviation
  from this document's snippets, not an oversight; `256` is the batch size every other DANINHAS run
  in this repo uses.
- **`hier_L2b.json`'s micro mirror**: the naive halving of `hier_L2a`'s micro `[8,4]` — `[4,2]` —
  triggers a pre-existing vendored `dtype=object` bug (`dalmax/tools/SSL/src/utils.py:28`, was
  `core/tools/SSL/src/utils.py:28` before Phase 4's move, documented
  in `dalmax/selection/hierarchical_kmeans.py`) when the real `DATA/daninhas_micro` embedding
  happens to split evenly at `--seed 1`; `[5,2]` was substituted after empirically probing several
  candidates against the actual derived selection RNG and embeddings (see
  `files_config/ablations/README.md` for the full account). Full-scale `hier_L2b.json` is unaffected
  (unchanged `[100,50]`) — an exact-even split is not a practical risk at the ~10k-image full pool.
- **`hier_L4.json`'s micro mirror**: used `[12,6,3,2]` rather than the naive `[8,4,2,1]` (bottoming
  out at a cluster of size 1), since the vendored pipeline's behavior at `n_clusters=1` is untested;
  this was verified empirically to run cleanly at `--seed 1` (see `files_config/ablations/README.md`).
- **`stage_full.json`**: materialized for pipeline consistency (so all three §6.3 rows share one
  code path through `RepresentationStrategy`), even though this document's own §6.3 says the "RNHAL
  (full)" row's *reported* number should come from the pre-Phase-2 `results/dalmax{1,2}/` reference
  runs, not a new run — both are valid sources; `run_ablation_gpu_1.sh` runs `stage_full.json` so a
  cross-check is available, but does not obsolete the reference-run recomputation path.

## Mapping to the paper's `\subsubsection`s

| This spec section | Paper subsubsection (working title) |
|---|---|
| §6.1 Representation ablation | "Representation ablation" / effect of spatial vs. spectral SSRAE components |
| §6.2 Hierarchy ablation | "Hierarchy ablation" / effect of hierarchy depth `L` and cluster counts `k_i` |
| §6.3 Contribution of the two RNHAL stages | "Contribution of the two RNHAL stages" / ablating `Φ` and `Γ` independently |

TBD: confirm exact subsubsection titles once the advisor finalizes the
`\subsection{Ablation study}` outline in `phd_files/Active_Learning_Mario/`.
