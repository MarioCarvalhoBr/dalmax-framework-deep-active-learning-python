# 00 — Overview

## What DalMax is

**DalMax** (kept as the project's single name — no rename) is a PhD research
laboratory for **Deep Active Learning applied to UAV weed recognition in
precision agriculture** (UFMS, Brazil, advisor: Wesley Nunes Gonçalves). It is
a Python research codebase, not a product: an entry-point script
(`demo.py`) drives an active-learning loop (label a small seed set, train a
classifier, query new samples, repeat) over a UAV weed-image dataset, and
compares acquisition ("query") strategies by their resulting classification
metrics.

## Scientific goal

The dissertation's main contribution is **RNHAL (Randomized Network-guided
Hierarchical Active Learning)**, described formally in
`phd_files/Active_Learning_Mario/method_full.tex` (§`subsec:proposed_rnhal`).
RNHAL replaces classifier-confidence-driven query selection with a
representation + hierarchy pipeline operating on the geometry of an embedding
space, aiming to reduce redundant sampling in visually dense regions under
small annotation budgets.

RNHAL operates in two stages:

1. **Randomized-network representation module** — a function `Φ(·)` maps
   each unlabeled image `x` to an embedding `z_x = Φ(x) ∈ ℝ^d`. In the
   current implementation `Φ` is the **SSRAE** (spatio-spectral randomized
   autoencoder) descriptor (see
   `phd_files/artigo-original-tecnica-ssrae-manuscript.pdf` and
   `core/tools/SSRAE/extractor.py`). For an RGB image, local 3×3
   neighborhoods per channel are used to fit closed-form randomized
   autoencoders (Tikhonov-regularized output layer, hidden size `Q`), whose
   output weight matrices are stretched into per-channel/per-pair
   descriptors and concatenated:
   `Φ(x) = [Θ_R(Q), Θ_G(Q), Θ_B(Q), Ω_RG(Q), Ω_GB(Q), Ω_BR(Q)]`
   — conceptually six equal-shape blocks: `Θ_R, Θ_G, Θ_B` are **spatial**
   signatures (within-channel), `Ω_RG, Ω_GB, Ω_BR` are **spectral**
   signatures (cross-channel). Verified in code
   (`core/tools/SSRAE/extractor.py:99`, variable names `beta_R, beta_G,
   beta_B, beta_S_R, beta_S_G, beta_S_B` for the same six blocks in that
   order, each block shaped `(9, Q+1)`). **Layout caveat (verified
   2026-08-23)**: the code concatenates these via `torch.hstack(...).reshape((1,
   -1))`, which is **row-interleaved**, not six contiguous chunks — naive
   slicing like `emb[:len(emb)//2]` does NOT recover the spatial-only
   signatures. See `.specs/experiments/ablation-study.md` §6.1 for the
   verified layout and the correct column-group slicing. Current hidden size
   **Q = 13**, hardcoded at `utils/data.py:139`. An alternative
   representation, **VCTex** (`core/tools/VCTex/`), uses `Q = [5, 17]`
   (`utils/data.py:70`).
2. **Hierarchical selection operator** — `Γ_{n_query}(·)` recursively
   partitions the embedding set `Z_U^(t) = {Φ(x) | x ∈ D_U^(t)}` via
   hierarchical k-means (`core/tools/SSL/src/hierarchical_kmeans_gpu.py`,
   `core/tools/SSL/src/hierarchical_sampling.py`) and samples across the
   resulting hierarchy to return the query batch `Q_t` of size `n_query`.
   Implemented as strategies `SSRAEKmeansHCSampling` / `VCTexKmeansHCSampling`
   in `core/query_strategies/ssl_ssrae_sampling.py`. Config is `config_kmh`
   in the params JSON: `n_clusters` (list, one entry per level), `n_levels`
   (int `L`), `sample_sizes` (list, samples drawn per level, one entry per
   level — call it `k_i` per level `i`).

Notation used across the specs (from `method_full.tex`):
`Φ` = representation function, `Γ_{n_query}` = hierarchical selection
operator, `Q` = SSRAE/VCTex hidden-neuron count (NOT to be confused with
`n_query`, the AL query budget — unfortunate name collision in the paper's
own notation vs. the codebase's CLI flag; be explicit about which `Q` is
meant everywhere in these specs), `L` = number of hierarchy levels
(`n_levels`), `k_i` = target sample count at hierarchy level `i`
(`sample_sizes[i]`).

A simpler, non-hierarchical variant, `SSRAEKmeansSampling` /
`VCTexKmeansSampling` (`core/query_strategies/ssrae_kmeans_sampling.py`,
`vctex_kmeans_sampling.py`), runs flat k-means with `n_clusters = n_query`
and picks, per cluster, the single sample closest to the centroid — this is
distinct from "hierarchical" and matters for the ablation in
`experiments/ablation-study.md` §6.3.

## Current status

- The paper under revision is `phd_files/Artigo_melhorias_Active_Learning_Mario.pdf`
  (LaTeX sources: `phd_files/Active_Learning_Mario/`).
- The advisor has requested an **ablation study** for the paper's
  `\subsection{Ablation study}` — see `.specs/experiments/ablation-study.md`.
  It is not yet implemented in code; the current codebase can only reproduce
  the reference RNHAL runs already executed (see
  `.specs/experiments/baseline-results.md`).
- Refactor **Phase 2 (core) is complete**: the `dalmax/` package makes the ablations
  one-config-line runs (`RepresentationStrategy` + `embedding`/`selection` params blocks).
  Phase 3 (ablation execution) is next.
  See `.specs/architecture/refactor-plan.md` (owned by a different work
  batch) for the phased plan; the target design is in
  `.specs/architecture/target-architecture.md`.
- Package layout `core/` + `utils/` is expected to eventually consolidate
  into a single `dalmax/` package (Phase 4 of the refactor plan) — recorded
  as an architecture decision, not yet executed.

## Pointers

- Method definition (LaTeX, source of truth for notation): `phd_files/Active_Learning_Mario/method_full.tex`
- Entry point: `demo.py` (thin shim) → `dalmax/cli.py` → `dalmax/experiment/runner.py`
- Strategy registry: `dalmax/query_strategies/registry.py` (legacy `utils/orchestrator.py` is dead code, Phase 4 removal)
- Dataset loading / feature caching: `utils/data.py`, `utils/dataset.py`
- Experimental protocol: `.specs/experiments/experimental-protocol.md`
- Ablation spec: `.specs/experiments/ablation-study.md`
- Where existing results live: `.specs/experiments/baseline-results.md`
- Execution environments: `.specs/infrastructure/execution-environments.md`
- Known issues: `.specs/quality/known-issues.md` (owned by a different work
  batch)
- Spec index and update contract: `.specs/README.md`
