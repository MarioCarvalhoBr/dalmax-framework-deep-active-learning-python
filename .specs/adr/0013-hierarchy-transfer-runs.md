# ADR 0013: Run the KMH-sweep winning hierarchy with the learned descriptors of papers 2 and 3

- **Status:** Accepted (author decision 2026-11-10), executed 2026-11-11 (UTC) and verified 2026-11-10 (local time)
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
- Follow-up (done): run it (`COLAB_RUNBOOK.md` section 6c), verify (`OK=222`), regenerate the report.

## Outcome (executed 2026-11-11 UTC, verified 2026-11-10 local)

All 12 runs ran in one Google Colab Pro session (start timestamps 2026-11-11 01:53 to 02:18 UTC, code commit `eee5d4e`). `make campaign-verify`: `OK=222 INCOMPLETE=0 MISSING=0 (of 222 jobs)`; seed audit PASS for seeds 1, 2 and 3 (73 runs compared per seed, reference `shared/random_nq100`); `make campaign-report` regenerated `docs/results/campaign` (74/74 groups with data). Runtime image (past-tense record): NVIDIA L4 GPU (22 GB), 12 vCPU Intel Xeon 2.20 GHz, 53 GB RAM, Ubuntu 24.04, Python 3.12.16, PyTorch 2.5.0+cu124, cuDNN 9.1, driver 580.82.07, Colab release `release-colab-external-images_20261109-060054_RC00`. This is a third runtime image of the campaign (the earlier 210 runs used two A100 images). All 12 runs logged the known `cumsum_cuda_kernel` non-deterministic-op warning (now 102 of 222 campaign runs).

Final round, seeds 1-3, mean +/- population sd:

| Group id | Representation, hierarchy | Weighted F1 (mean +/- sd) | Macro F1 | Accuracy | Seeds 1 / 2 / 3 (weighted F1) |
|---|---|---|---|---|---|
| `rnhal/6_2/hier_L2_matched__no_resampling` | SSRAE q=13, `[600,100]`, `[30,5]`, m=0 | 0.8592 +/- 0.0138 | 0.8054 +/- 0.0141 | 0.8506 | 0.8538 / 0.8457 / 0.8782 |
| `rnhal/6_2/hier_L3__no_resampling` | SSRAE q=13, `[600,200,100]`, `[30,15,2]`, m=0 | 0.8740 +/- 0.0217 | 0.8276 +/- 0.0272 | 0.8677 | 0.8444 / 0.8960 / 0.8817 |
| `texhal_q17/6_2/hier_L2_matched__no_resampling` | VCTex Q=17, `[600,100]`, `[30,5]`, m=0 | 0.8544 +/- 0.0244 | 0.8062 +/- 0.0276 | 0.8424 | 0.8260 / 0.8855 / 0.8519 |
| `texhal_q17/6_2/hier_L3__no_resampling` | VCTex Q=17, `[600,200,100]`, `[30,15,2]`, m=0 | 0.8642 +/- 0.0216 | 0.8178 +/- 0.0247 | 0.8561 | 0.8380 / 0.8909 / 0.8637 |

Reference values: RNHAL reference (m=10) 0.8805 +/- 0.0084; RNHAL L=4 `[800,600,200,100]` 0.8955; TexHAL Q=17 reference (m=10)
0.8745 +/- 0.0079; TexHAL best `[300,100,50]` 0.8842; KMH (ImageNet features) `k=[600,100]` without resampling 0.8819 +/- 0.0148
(KMH sweep stage 2 winner), KMH reference shape without resampling 0.8525, KMH reference with m=10 (`shared/kmh_nq100`) 0.8224.

Reading (fixed before the runs in the thesis plan): the KMH-winner shape does NOT transfer to the descriptors. Under
`k=[600,100]` without resampling both descriptors are below the ImageNet features (SSRAE 2.3 points below, VCTex 2.8 below,
beyond one sd). Turning resampling off at the reference shape costs the descriptors 0.7 (SSRAE) and 1.0 (VCTex) points, within
one sd, and roughly triples the spread across seeds, whereas it had gained 3.0 points for the ImageNet features. Hierarchy and
representation interact: the descriptors' advantage over ImageNet features holds under the reference and deeper hierarchies,
not under the hierarchy tuned for ImageNet features. No canonical configuration changes (author decision 2026-11-10: the
reference hierarchy stays canonical for RNHAL and TexHAL).

