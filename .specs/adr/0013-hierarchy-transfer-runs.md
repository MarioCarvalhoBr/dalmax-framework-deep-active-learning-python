# ADR 0013: Run the KMH-sweep winning hierarchy with the learned descriptors of papers 2 and 3

- **Status:** Accepted (author decision 2026-11-10), not executed
- **Date:** 2026-11-10

## Context

The KMH parameter sweep (ADR 0012), stage 2, found that the budget-matched two-level hierarchy
`n_clusters=[600,100]`, `sample_sizes=[30,5]`, `n_resamples=0` over frozen ImageNet ResNet-50 features reaches
**0.8819** final-round weighted F1 at `n_query=100` (seeds 1-3), above the reference hierarchy of RNHAL (0.8805,
SSRAE `Q=13`, `[600,200,100]`) and of TexHAL (0.8745, VCTex `Q=17`); see `.specs/experiments/campaign.md`
("Stage 2 results"). That sweep varies the selection only with the ImageNet descriptor, so it cannot say
whether the shape (two levels, budget-matched) or the knob (no centroid resampling) is responsible, nor whether
the gain transfers to the learned descriptors. The three papers must be comparable at that configuration, and
the "no resampling" knob has to be separated from the shape.

## Decision

We will add ONE part, `hier_transfer`, to the main campaign manifest (`files_config/campaign/manifest.json`, micro
mirror `manifest_micro.json`): four groups, seeds 1-3, `n_query=100`, `n_init_labeled=100`, `n_round=8`,
`RepresentationStrategy`, the same training recipe as the existing 6.2 hierarchy rows of each method:

| Group id | Params | Hierarchy |
|---|---|---|
| `rnhal/6_2/hier_L2_matched__no_resampling` | `files_config/ablations/rnhal/hier_L2_matched__no_resampling.json` | `[600,100]`, `[30,5]`, `n_resamples=0` |
| `rnhal/6_2/hier_L3__no_resampling` | `files_config/ablations/rnhal/hier_L3__no_resampling.json` | `[600,200,100]`, `[30,15,2]`, `n_resamples=0` |
| `texhal_q17/6_2/hier_L2_matched__no_resampling` | `files_config/ablations/texhal_q17/hier_L2_matched__no_resampling.json` | `[600,100]`, `[30,5]`, `n_resamples=0` |
| `texhal_q17/6_2/hier_L3__no_resampling` | `files_config/ablations/texhal_q17/hier_L3__no_resampling.json` | `[600,200,100]`, `[30,15,2]`, `n_resamples=0` |

The RNHAL rows use SSRAE `q=13` and the TexHAL rows VCTex `q=[17]` (ADR 0011). The `L=3` rows keep the reference
`sample_sizes=[30,15,2]` (the sweep's `hier_L3_ref`), so the only difference from the reference runs is
`n_resamples=0`. The groups are rows appended to the 6.2 hierarchy table of paper 3 (RNHAL) and paper 2 (TexHAL
Q=17 table), labelled `L=2, k=[600,100], no resampling (KMH sweep winner)` and `L=3, k=[600,200,100] (reference),
no resampling`. The part sits after `texhal_q17` in `PARTS` and in the execution order, before the
`control`/KMH sweep parts of the other manifests; `upper_bound` stays last. The redundancy checker confirms none of
the four duplicates an existing computation (the sweep's `kmh/nq100/hier_L2_matched__no_resampling` uses the
ImageNet extractor).

## Consequences

- The main campaign grows from 210 runs / 70 groups to **222 runs / 74 groups** (12 new runs); the 210 executed runs
  are untouched and re-running `make campaign-run PART=all` resumes and executes only the 12 new ones.
- Estimated cost: about 2 to 4 GPU hours on Colab (the `L=2` shape clusters in a few seconds per round; the `L=3`
  shape costs about 1 minute per round).
- The results feed table 6.2 of papers 2 and 3 and the thesis chapter 9; the comparison tables are unchanged, so
  which hierarchy becomes canonical for RNHAL/TexHAL remains an open author/advisor decision.
- Disk: about 1.1 GB more checkpoints (12 x ~95 MB) under `results/campaign/`.
- Follow-up: run it (`COLAB_RUNBOOK.md` section 6c), verify (`OK=222` once everything is present), regenerate the report.
