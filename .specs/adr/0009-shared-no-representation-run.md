# ADR 0009: One shared "w/o representation module" run for papers 2 and 3 (and paper 1's KMH@100)

- **Status:** Accepted
- **Date:** 2026-09-29
- **Supersedes:** the 2026-09-01 decision (recorded in `.specs/experiments/papers-roadmap.md` and
  `.specs/experiments/ablation-study-texhal.md`) that `stage_no_representation` is run separately
  for each paper.

## Context

`files_config/ablations/rnhal/stage_no_representation.json` and
`files_config/ablations/texhal/stage_no_representation.json` (both now removed) were byte-for-byte identical: the
ImageNet ResNet-50 embedding (`resnet_imagenet`, `q: null`) plus hierarchical k-means with the
reference hierarchy `[600,200,100]` / `[30,15,2]` involves neither SSRAE nor VCTex. On 2026-09-01
the user decided each paper would nevertheless run its own copy. Running it twice costs GPU time
and yields two different numbers for the same computation across two papers. In the 2026-09-29
scope change, paper 1 also gained a "KMH" baseline (hierarchical k-means over ImageNet ResNet-50
features, no SSRAE/VCTex), whose n_query=100 instance is the same computation as well.

## Decision

We will run this configuration **once** per seed. In `files_config/campaign/manifest.json` it is the
single group `shared/kmh_nq100` (config: the neutral `files_config/campaign/params_kmh.json`),
consumed by paper 1 (`p1/KMH/nq100`, the KMH row), paper 2's §6.3 "w/o representation module" row and
paper 3's §6.3 row (`texhal/6_3/stage_no_representation` and `rnhal/6_3/stage_no_representation` are
aliases of it). Both ablation copies (`rnhal/` and `texhal/` `stage_no_representation.json`, plus their
micro mirrors) were removed so no copy of the config lives outside `files_config/campaign/`. The
ablation-only scripts no longer run it; `ablation_report` and `results_doctor` resolve the texhal row
to the rnhal results tree (where the legacy 2026-08-26 result lives).

## Consequences

- Papers 1, 2 and 3 report the identical value for this row; one fewer config (11 texhal configs).
- The 2026-09-01 "not shared" decision is superseded (kept in the docs, marked as such).
- The `duplicate by explicit decision` exemption of the redundancy checker no longer exists.
- Cross-check "the two papers' runs agree within GPU noise" is moot (there is one run).
