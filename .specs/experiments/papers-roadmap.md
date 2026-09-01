# Papers roadmap

Three-paper plan for this PhD's active-learning-for-UAV-weed-recognition work, as stated by the
user (Mário Carvalho) and recorded here per `.claude/rules/spec-sync.md` ("new rules or conventions
that emerge in conversation must be persisted"). This is the authoritative index of what each paper
covers and how the codebase's ablation suites map onto them — `files_config/ablations/README.md`,
`.specs/experiments/ablation-study.md` (RNHAL), and `.specs/experiments/ablation-study-texhal.md`
(TexHAL) all cross-reference this file.

## The three papers

| # | Working title / scope | Representation | Selection | Ablation suite | Status |
|---|---|---|---|---|---|
| 1 | DAL benchmark comparison | n/a (all non-representation baselines) | n/a | none | first to publish; establishes the baseline comparison table other papers beat |
| 2 | TexHAL proposal | VCTex (color-texture, RAE-based, Fares & Ribas) | Hierarchical k-means | `files_config/ablations/texhal/` | proposal + ablations show it beats paper 1's baselines |
| 3 | RNHAL proposal | SSRAE (randomized-network spatio-spectral) | Hierarchical k-means | `files_config/ablations/rnhal/` | proposal + ablations show it beats TexHAL and all paper-1 baselines; **ablations executed** 2026-08-26 |

**Paper 1 — DAL benchmark comparison.** A systematic comparison of existing deep active learning
query strategies on the `daninhas_full` UAV weed dataset: `RandomSampling`, `LeastConfidence`,
`MarginSampling`, `EntropySampling`, the `*Dropout` variants, `KMeansSampling`, `KCenterGreedy`,
`BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool` — i.e. every strategy in
`dalmax/query_strategies/registry.py::STRATEGY_REGISTRY` that is **not** a representation-based
preset (no SSRAE, no VCTex). This paper establishes the baseline comparison table that papers 2 and
3 each need to beat. Intended to be the first of the three published, since papers 2 and 3's claims
("beats the baselines") depend on this table existing. No ablation suite of its own — it is a
benchmark sweep (`files_config/benchmark/params_df_gpu_{0,1}.json`,
`scripts/benchmark/run_pipe_gpu_{0,1}.sh`), not an ablation study.

**Paper 2 — TexHAL.** Proposes **TexHAL** (**Tex**ture-guided **H**ierarchical **A**ctive
**L**earning): the VCTex color-texture representation (Ricardo T. Fares and Lucas C. Ribas, São
Paulo State University / UNESP, "Volumetric Color-Texture Representation for Colorectal Polyp
Classification in Histopathology Images," `phd_files/artigo-original-tecnica-vctex-manuscript.pdf`)
combined with the same hierarchical k-means batch-selection module RNHAL uses. Claim: TexHAL beats
paper 1's baselines. CLI preset (unchanged by this naming): `VCTexKmeansHCSampling`. Ablation suite:
`files_config/ablations/texhal/` (`.specs/experiments/ablation-study-texhal.md`) — materialized
2026-08-30, not yet executed.

**Name alternatives considered for paper 2**: alternatives `VTHAL`, `VCT-HAL` were considered but rejected
(2026-09-01); **TexHAL** is the confirmed name (confirmed 2026-09-01 by user). The CLI preset name
`VCTexKmeansHCSampling` and every `files_config/ablations/texhal/*.json` file are unaffected by the
paper's working title either way, since they name the method's mechanism — VCTex + hierarchical —
not the paper.

**Paper 3 — RNHAL.** Proposes **RNHAL** (randomized-network spatio-spectral representation, SSRAE,
plus the same hierarchical k-means selection module). Claim: RNHAL beats both TexHAL (paper 2) and
every paper-1 baseline. CLI preset: `SSRAEKmeansHCSampling`. Ablation suite:
`files_config/ablations/rnhal/` (`.specs/experiments/ablation-study.md`) — **executed** 2026-08-26
on Google Colab Pro (33/33 runs, zero failures); this is the main contribution described in
`CLAUDE.md`'s "What this project is."

**Future (out of scope for now)**: multimodal models — combining the UAV imagery with other
sensing modalities — is a noted future direction, not part of the current three-paper plan. No
further detail is tracked here until it becomes active work.

## How this maps onto the shared codebase

All three papers/methods share one codebase (`dalmax/`) via the registry pattern
(`.claude/rules/code-quality.md`): `RepresentationStrategy` + the `EmbeddingProvider`/
`SelectionStrategy` abstractions make "TexHAL vs. RNHAL" a config-only difference (which
`embedding.extractor` is selected), not a fork of the training/selection loop. Only the **ablation
suites** (which sweep each method's own representation/hierarchy/stage axes) are organized
per-method on disk, under `files_config/ablations/{rnhal,texhal}/` — see that folder's `README.md`
for the full naming-convention rationale (why §6.1's rows differ between the two methods but
§6.2/§6.3 don't).

Results and reporting follow the same per-method split: `results/ablations/{rnhal,texhal}/` (new
runs; the already-executed RNHAL batch's legacy root, `results/ablations/{6_1,6_2,6_3}/` with no
method segment, is append-only and left in place) and
`docs/results/ablation_tables/{rnhal,texhal}/` (new committed tables; the RNHAL batch's already-
committed top-level `docs/results/ablation_tables/*.{csv,md,tex}` files are unaffected — see
`docs/results/README.md`).

## Confirmed decisions (2026-09-01)

- **TexHAL name**: confirmed as **TexHAL** (alternatives `VTHAL`, `VCT-HAL` rejected 2026-09-01).
- **§6.1 reference hierarchy for TexHAL**: confirmed to match RNHAL's reference hierarchy
  (`n_clusters=[600,200,100]`, `n_levels=3`, `sample_sizes=[30,15,2]`) for cross-paper comparability
  (confirmed 2026-09-01).
- **`stage_no_representation` sharing**: confirmed NOT shared — papers 2 and 3 each run their own
  independently (confirmed 2026-09-01), even though the config is byte-for-byte identical between the two methods.
