# 00 — Overview

## What DalMax is

**DalMax** (kept as the project's single name — no rename) is a PhD research
laboratory for **Deep Active Learning applied to UAV weed recognition in
precision agriculture** (UFMS, Brazil, advisor: Wesley Nunes Gonçalves). It is
a Python research codebase, not a product: an entry-point script
(`trainer.py`, renamed from `demo.py` on 2026-08-23 — see ADR 0006; the historical name `demo.py` no longer exists) drives an active-learning loop (label a small seed set, train a
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
   `dalmax/tools/SSRAE/extractor.py`). For an RGB image, local 3×3
   neighborhoods per channel are used to fit closed-form randomized
   autoencoders (Tikhonov-regularized output layer, hidden size `Q`), whose
   output weight matrices are stretched into per-channel/per-pair
   descriptors and concatenated:
   `Φ(x) = [Θ_R(Q), Θ_G(Q), Θ_B(Q), Ω_RG(Q), Ω_GB(Q), Ω_BR(Q)]`
   — conceptually six equal-shape blocks: `Θ_R, Θ_G, Θ_B` are **spatial**
   signatures (within-channel), `Ω_RG, Ω_GB, Ω_BR` are **spectral**
   signatures (cross-channel). Verified in code
   (`dalmax/tools/SSRAE/extractor.py` — was `core/tools/SSRAE/extractor.py:99`
   before Phase 4's move — variable names `beta_R, beta_G,
   beta_B, beta_S_R, beta_S_G, beta_S_B` for the same six blocks in that
   order, each block shaped `(9, Q+1)`). **Layout caveat (verified
   2026-08-23)**: the code concatenates these via `torch.hstack(...).reshape((1,
   -1))`, which is **row-interleaved**, not six contiguous chunks — naive
   slicing like `emb[:len(emb)//2]` does NOT recover the spatial-only
   signatures. See `.specs/experiments/ablation-study.md` §6.1 for the
   verified layout and the correct column-group slicing. `Q` is now
   config-driven (`EmbeddingConfig.q`, `dalmax/config/loader.py`), defaulting
   to the historical values: **Q = 13** for SSRAE (was hardcoded at
   `utils/data.py:139`, that file deleted in Phase 4) and `Q = [5, 17]` for
   the alternative representation **VCTex** (`dalmax/tools/VCTex/`, was
   hardcoded at `utils/data.py:70`).
2. **Hierarchical selection operator** — `Γ_{n_query}(·)` recursively
   partitions the embedding set `Z_U^(t) = {Φ(x) | x ∈ D_U^(t)}` via
   hierarchical k-means (`dalmax/tools/SSL/src/hierarchical_kmeans_gpu.py`,
   `dalmax/tools/SSL/src/hierarchical_sampling.py`) and samples across the
   resulting hierarchy to return the query batch `Q_t` of size `n_query`.
   The CLI names `SSRAEKmeansHCSampling` / `VCTexKmeansHCSampling` are served
   by `dalmax.query_strategies.representation.RepresentationStrategy` +
   `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`
   (the fixed-behavior classes that previously implemented this,
   `core/query_strategies/ssl_ssrae_sampling.py`, were deleted in Phase 4).
   Config is `config_kmh` in the params JSON: `n_clusters` (list, one entry
   per level), `n_levels` (int `L`), `sample_sizes` (list, samples drawn per
   level, one entry per level — call it `k_i` per level `i`).

Notation used across the specs (from `method_full.tex`):
`Φ` = representation function, `Γ_{n_query}` = hierarchical selection
operator, `Q` = SSRAE/VCTex hidden-neuron count (NOT to be confused with
`n_query`, the AL query budget — unfortunate name collision in the paper's
own notation vs. the codebase's CLI flag; be explicit about which `Q` is
meant everywhere in these specs), `L` = number of hierarchy levels
(`n_levels`), `k_i` = target sample count at hierarchy level `i`
(`sample_sizes[i]`).

A simpler, non-hierarchical variant, `SSRAEKmeansSampling` /
`VCTexKmeansSampling` — served by `dalmax/selection/flat_kmeans_closest.py::FlatKMeansClosest`
(the classes that previously implemented these directly, `core/query_strategies/
ssrae_kmeans_sampling.py`/`vctex_kmeans_sampling.py`, were deleted in Phase 4) —
runs flat k-means with `n_clusters = n_query`
and picks, per cluster, the single sample closest to the centroid — this is
distinct from "hierarchical" and matters for the ablation in
`experiments/ablation-study.md` §6.3.

## Current status

- The paper under revision is `phd_files/Artigo_melhorias_Active_Learning_Mario.pdf`
  (LaTeX sources: `phd_files/Active_Learning_Mario/`).
- **Implementation findings, accepted as-is (2026-11-02, author decision)**: the ResNet-50 model has no dropout (the 4 MC-dropout/BALD strategies are degenerate, KI-41) and `KMeansSampling`/`KCenterGreedy` use an untrained random embedding (KI-42); paper-1 numbers are reported as executed with disclosure, see `quality/known-issues.md`. **Reversed 2026-11-11 (ADR 0014)**: KI-41, KI-42 and KI-45 (AdversarialBIM ties) are fixed in code and the 63 affected runs are prepared as campaign parts `paper1_rerun_a`/`paper1_rerun_b` (not executed).
- **Clean re-run prepared (2026-11-11, ADR 0016, not executed)**: the first campaign's dataset `DATA/daninhas_full` has
  byte-identical duplicates inside and across train and test (KI-46: 171 distinct images in both splits). Author
  decision: EVERYTHING is re-run on the duplicate-free `DATA/daninhas_full_fixed` (7,230 train / 1,008 test), with the
  eight DeepAL+ strategies of ADR 0015 added to paper 1 (20 strategies + KMH x 3 budgets) and, since 2026-11-12, the
  WAAL training-budget control `WAALRandomQuery` at nq100 (3 runs, a separate control row): main manifest 297 runs + KMH
  sweep 165 runs into `results/campaign_fixed/` and `results/campaign_kmh_fixed/`, split into two GPU parts for two
  Colab accounts (about 99 estimated GPU hours in total; `COLAB_RUNBOOK.md` section 6). Every number in the next bullet
  is history of the first campaign and is superseded for the papers by this re-run.
- **Clean campaign executed; TexHAL canonical `Q=13` (2026-11-15, ADR 0017)**: the 462 runs above were executed, merged
  locally and verified (OK=462, one commit = tag `campaign-fixed-v2`, seed and data-order audits PASS). From its paper-2
  6.1 table (nq100 weighted F1: VCTex `Q=13` 0.8341, `Q=[5,17]` 0.8038, `Q=5` 0.7766, `Q=17` 0.7703; KMH 0.7921) the
  author made `Q=13` TexHAL's canonical configuration (superseding ADR 0011's `Q=17`; Q chosen on the test set,
  disclosed). Its 6.2/6.3 ablations are the new part `texhal_q13` (8 groups / 24 runs in `part_1_gpu`, prepared, not
  executed; tag `campaign-fixed-v3`; `COLAB_RUNBOOK.md` section 6e): main manifest 321 runs / 107 groups, 486 in total
  with the sweep.
- **KMH wide-top exploration (2026-11-15, ADR 0018)**: paper 1's KMH is below WAAL at every query size on the clean
  campaign (best KMH sweep configuration vs WAAL: nq100 0.7921 vs 0.8077, nq50 0.7146 vs 0.7874, nq10 0.5627 vs
  0.5793). New part `kmh_wide` of the KMH sweep (13 groups / 39 runs in `part_2_gpu`, prepared, not executed; tag
  `campaign-fixed-v4`; `COLAB_RUNBOOK.md` section 6f): wide tops ending at the query size and one-level `k=1200`/`2400`
  hierarchies. Sweep 204 runs / 68 groups, `part_2_gpu` 216 / 72, 525 runs in total. A configuration chosen from it is
  selected on the test set and must be confirmed on seeds 4-6 with RandomSampling and WAAL before any claim (planned).
- **KMH feature post-processing (2026-11-15, ADR 0019)**: new optional params key `embedding.postprocess`
  (`none` default = identity, `l2`, `zscore`, `pca_l2` + `pca_dim`; `dalmax/embeddings/postprocess.py`, fitted once on
  the initial unlabeled pool, applied by `RepresentationStrategy` before the selection; the default path is tested
  identical to tag `campaign-fixed-v4`). New part `kmh_norm` of the KMH sweep (9 groups / 27 runs in `part_1_gpu`,
  prepared, not executed; tag `campaign-fixed-v5`, which unlike v3/v4 changes trainer code behind the new key;
  `COLAB_RUNBOOK.md` section 6g): the reference hierarchy with the ImageNet features post-processed by `l2`, `zscore`
  and `pca_l2` (64) at nq 10/50/100. Sweep 231 runs / 77 groups, `part_1_gpu` 336 / 112, 552 runs in total. Same
  confirmation rule on seeds 4-6 before any claim.
- **Descriptors at small query sizes and seeds 4-6 (2026-11-16, ADR 0020)**: two config-only parts of the main manifest,
  both in `part_2_gpu` (account B, tag `campaign-fixed-v6`, config-only relative to v5; `COLAB_RUNBOOK.md` section 6h;
  prepared, not executed): `descriptors_nq` (RNHAL and TexHAL in their executed nq100 configurations at n_query 50 and
  10, 4 groups / 12 runs) and `seeds_ext` (the 8 headline methods at nq100 on the new seeds 4, 5, 6, 8 groups / 24
  runs). Main manifest 357 runs / 119 groups, `part_2_gpu` 252 / 84, 588 runs in total. The campaign engine now
  treats two groups as redundant only on a common seed, and picks the pilot only among groups that run seed 1.
- **KMH l2 confirmation on seeds 4-6 (2026-11-16, ADR 0021)**: `kmh_norm` executed and verified (552 OK); its `l2`
  variant (fixed before any seed 4-6 result) is confirmed by the config-only part `kmh_confirm` of the main manifest,
  `part_1_gpu` (account A, tag `campaign-fixed-v7`, config-only relative to v6; `COLAB_RUNBOOK.md` section 6i; prepared,
  not executed): KMH `l2` at nq 100/50/10, the KMH reference, RandomSampling and WAAL at nq 50/10, on seeds 4, 5, 6
  (9 groups / 27 runs; the nq100 comparison rows are the `seeds_ext` runs). Main manifest 384 runs / 128 groups,
  `part_1_gpu` 363 / 121, 615 runs in total.
- **Pre-flight fix KI-47 (2026-11-12, ADR 0016 amendment)**: the loaders listed each class folder in file-system order,
  which differed between Colab sessions, so in the first campaign the same seed started from other images in each
  later session and the cached embeddings served later sessions the features of other images (the KMH sweep,
  `texhal_q17`, `hier_transfer`, paper-1 KMH nq50/nq10; very likely the cause of KI-43). Fixed before the clean
  re-run: sorted file order, `data_order` hashes in `run_metadata.json`, an order-keyed and path-validated cache, the
  seed audit by image paths plus a data-order audit; also a Drive quota gate and a VAAL/WAAL pilot.
- **(History) Campaign executed (2026-09-30 to 2026-10-05, verified 2026-11-01; part `texhal_q17` executed 2026-11-01, verified 2026-11-02: 210/210)**: all runs of the three
  papers (one manifest, 3 seeds, Google Colab Pro) completed, 0 incomplete/missing, seed audit PASS;
  numbers in `.specs/experiments/campaign.md` (execution record), reports in `docs/results/campaign/`
  (local-only, gitignored). Headline (nq=100, weighted F1): upper bound 0.9157; RNHAL 0.8805 (best
  hierarchy `L=4 [800,600,200,100]`: 0.8955); best classical method LeastConfidence 0.8716; TexHAL
  (Q=[5,17]) 0.8259, TexHAL canonical Q=17 0.8745 (KI-40, ADR 0011: accepted by the author 2026-11-01, advisor review pending; its 18-run `texhal_q17` re-run of 6.2/6.3 executed 2026-11-01, campaign now 210/210 runs). **Current phase:
  thesis writing** (started 2026-11-01 under `thesis_drafts/`, gitignored).
- (History) The advisor has requested an **ablation study** for the paper's
  `\subsection{Ablation study}` — see `.specs/experiments/ablation-study.md`.
  The 11 run-table rows were first executed on 2026-08-26 (Colab, history) and
  re-executed authoritatively in the campaign (see above;
  `.specs/experiments/ablation-study.md`, `ablation-study-texhal.md`).
- Refactor **Phases 1-4 are complete (Phase 3 ablations executed 2026-08-26 and again, authoritatively, in the campaign)**: the `dalmax/` package
  makes the ablations one-config-line runs (`RepresentationStrategy` +
  `embedding`/`selection` params blocks). See `.specs/architecture/refactor-plan.md`
  (owned by a different work batch) for the phased plan; the target design is in
  `.specs/architecture/target-architecture.md`.
- Package layout `core/` + `utils/` has been fully consolidated into a single
  `dalmax/` package (Phase 4 of the refactor plan, executed 2026-08-23) —
  `core/` and `utils/` no longer exist on disk. See ADR 0002's final
  amendment and `.specs/architecture/current-state.md`.

## Pointers

- Method definition (LaTeX, source of truth for notation): `phd_files/Active_Learning_Mario/method_full.tex`
- Entry point: `trainer.py` (historical name `demo.py`; thin shim) → `dalmax/cli.py` → `dalmax/experiment/runner.py`
- Strategy registry: `dalmax/query_strategies/registry.py` (legacy `utils/orchestrator.py` was deleted in Phase 4)
- Dataset loading / feature caching: `dalmax/data/datasets.py`, `dalmax/data/handlers.py`, `dalmax/data/loaders.py`
- Experimental protocol: `.specs/experiments/experimental-protocol.md`
- Ablation spec: `.specs/experiments/ablation-study.md`
- Where existing results live: `.specs/experiments/baseline-results.md`
- Execution environments: `.specs/infrastructure/execution-environments.md`
- Known issues: `.specs/quality/known-issues.md` (owned by a different work
  batch)
- Spec index and update contract: `.specs/README.md`
