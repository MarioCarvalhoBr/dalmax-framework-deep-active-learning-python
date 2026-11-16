# ADR 0021: Confirm KMH with l2-normalised features on seeds 4-6, against the KMH reference, RandomSampling and WAAL (part `kmh_confirm`, account A)

- **Status:** Accepted (orchestrator brief of 2026-11-16, the confirmation ADR 0019 requires; thesis pendencia P42
  stage 3); prepared, not executed
- **Date:** 2026-11-16

## Context

ADR 0019 added the optional `embedding.postprocess` step and ran the part `kmh_norm` (the reference KMH hierarchy
`k=[600,200,100]`, `sample_sizes=[30,15,2]`, ImageNet ResNet-50 features, post-processed by `l2`, `zscore` or
`pca_l2` with 64 dimensions, at `n_query` 100, 50 and 10, seeds 1-3, account A, tag `campaign-fixed-v5`). It was
executed and verified (both manifests: 552 OK). Final-round weighted F1, mean of seeds 1-3 (population standard
deviation where it matters):

| `n_query` | KMH `l2` | KMH `pca64_l2` | KMH reference (raw) | WAAL | RandomSampling |
|---|---|---|---|---|---|
| 100 | 0.8128 +- 0.0171 | 0.8155 +- 0.0346 | 0.7921 | 0.8077 | 0.7397 |
| 50 | 0.7086 | 0.5589 +- 0.3422 (one seed collapsed) | 0.5860 | 0.7874 | 0.6455 |
| 10 | 0.4493 | 0.5575 | 0.4909 | 0.5793 | 0.4756 |

Every number of this table was read on the TEST set (there is no validation split) and chosen among three variants
per query size. ADR 0019 ("Selection on the test set") therefore requires a confirmation on seeds 4-6 with
RandomSampling and WAAL before any claim about the post-processing.

Seeds 4-6 exist only in the part `seeds_ext` (ADR 0020, account B, tag `campaign-fixed-v6`, running at the time of
this ADR), at `n_query = 100`: among its 8 groups, `seeds_ext/KMH` (the raw-feature reference), `seeds_ext/RandomSampling`
and `seeds_ext/WAAL`. No seed 4-6 result of any group exists when this ADR is written. Account A (`part_1_gpu`) is
idle.

## Decision

### The variant is fixed now: `l2` only

Before any seed 4-6 result exists, the variant to confirm is **`l2`** (`{"method": "l2"}`), at all three query
sizes:

- it is the simplest variant (no fitted statistic beyond each vector's own norm), and it was stable in every seed;
- it is best or tied at `n_query` 100 (0.8128 vs `pca64_l2` 0.8155, inside one standard deviation) and best at 50;
- `pca64_l2` is NOT confirmed: one of its three nq50 seeds collapsed (0.5589 +- 0.3422), an unstable variant;
- `zscore` is not confirmed;
- `l2` is confirmed at `n_query = 10` too, although it lost there (0.4493 against the reference 0.4909 and
  `pca64_l2` 0.5575): the choice is made once, for the variant, not per query size on the test set, and the nq10
  result is reported whatever it is.

### Part `kmh_confirm` (main manifest, account A)

A new part **`kmh_confirm`** of the MAIN manifest (`files_config/campaign/manifest.json`, generator
`scripts/campaign/build_manifest.py`, micro mirror `manifest_micro.json`): 9 run groups x seeds `[4, 5, 6]` = 27
runs, `gpu_part = "part_1_gpu"` (account A). Each group is an exact copy of an executed seeds-1-3 group (same params
file, strategy, `n_query`, `n_init_labeled` 100, `n_round` 8 and `gpu_stage`; the resolved config and the effective
signature are equal, tested); only the seeds, the GPU part and the results folder differ:

| group id | counterpart (seeds 1-3) | `gpu_stage` | estimate (s/run) |
|---|---|---|---|
| `kmh_confirm/nq100/kmh_l2` | `kmh_norm/nq100/l2` (sweep manifest) | 3 | 592 |
| `kmh_confirm/nq50/kmh_l2` | `kmh_norm/nq50/l2` (sweep manifest) | 3 | 559 |
| `kmh_confirm/nq10/kmh_l2` | `kmh_norm/nq10/l2` (sweep manifest) | 3 | 537 |
| `kmh_confirm/nq50/kmh_reference` | `p1/KMH/nq50` (`params_kmh.json`) | 1 | 574 |
| `kmh_confirm/nq10/kmh_reference` | `p1/KMH/nq10` | 1 | 543 |
| `kmh_confirm/nq50/RandomSampling` | `p1/RandomSampling/nq50` | 1 | 82 |
| `kmh_confirm/nq10/RandomSampling` | `p1/RandomSampling/nq10` | 1 | 56 |
| `kmh_confirm/nq50/WAAL` | `p1/WAAL/nq50` | 5 | 3,589 |
| `kmh_confirm/nq10/WAAL` | `p1/WAAL/nq10` | 5 | 3,650 |

The `l2` params files are those of `kmh_norm`
(`files_config/ablations/kmh_sweep/nq{N}/hier_L3_ref__norm_l2.json`); no params file is new.

**The nq100 reference, RandomSampling and WAAL are not duplicated.** On seeds 4-6 at `n_query = 100` they are the
`seeds_ext` groups (account B), the same computation on the same seeds. `kmh_confirm` has no group for them; the
nq100 table reads them through three aliases, `kmh_confirm/nq100/{kmh_reference,RandomSampling,WAAL}` ->
`seeds_ext/{KMH,RandomSampling,WAAL}` (the manifest validation checks that each alias has its canonical group's
effective signature). The redundancy validation (`find_redundant_groups`, "same computation on a common seed", ADR
0020) has nothing to object to: every `kmh_confirm` group shares its signature only with seeds-1-3 groups (tested).

### Why the main manifest, and why the tables are in paper 3

The KMH sweep manifest would be the natural home of a KMH round, but three constraints force the main manifest:

1. a table row must name a group or alias OF ITS OWN manifest (`validate_manifest`), and the nq100 table needs the
   `seeds_ext` rows, which live in the main manifest;
2. RandomSampling and WAAL are paper-1 strategies of the main manifest; the sweep manifest holds
   `RepresentationStrategy` groups only;
3. one zip folder per part (`ZIP_ONLY_PART`, ADR 0017): the 9 groups share `results/campaign_fixed/kmh_confirm/`.
   Splitting them over the two manifests would split them over two results roots.

Results: `results/campaign_fixed/kmh_confirm/nq{N}/<method>/` (`<method>` = `kmh_l2`, `kmh_reference`,
`RandomSampling`, `WAAL`; leaves `.../daninhas_full_fixed/SEED_{4,5,6}/NQ_..._NIL_100_NR_8_NE_.../<strategy>/`).

The report tables `kmh_confirm_nq100`, `kmh_confirm_nq50`, `kmh_confirm_nq10` (kind `benchmark`; rows KMH `l2`,
KMH reference, RandomSampling, WAAL, each aggregating its own group's seeds 4-6) are in **paper 3's** report, next
to `seeds_ext_nq100`. The single-consumer rule (a group consumed by more than one paper lives under `shared/`,
ADR 0009) is then kept without exemption: the three `seeds_ext` groups the nq100 table aliases are already consumed
by paper 3 only, and moving them under `shared/` would change the results folder of runs in progress on account B.
The paper folder of a report is bookkeeping: the KMH confirmation is a paper-1 question (paper 1's KMH row,
ADR 0019) and the thesis may cite the tables from any chapter. No seeds-1-3 table, row or `best_of` candidate reads
a `kmh_confirm` run (tested); seeds 4-6 are never merged into a seeds-1-3 number.

### Execution order and the ImageNet feature cache

Inside `part_1_gpu` (every earlier group is executed, so the plan skips them) the 27 new jobs run by stage: the KMH
reference and RandomSampling at nq50/nq10 (stage 1), the three `l2` groups (stage 3), WAAL (stage 5, last). Account
A's Drive has no ImageNet feature cache for seeds 4-6 (account B computes its own for `seeds_ext/KMH`; Drives are not
shared). The cache key (`dalmax/embeddings/cache.py`: dataset, extractor, Q, variant, split, order hash, pool hash)
does not depend on `n_query` (the pool hash is over the INITIAL unlabeled pool, fixed by the seed and
`n_init_labeled`), and the post-processing is applied after the cache load and never cached (ADR 0019): so the cache
of each seed is computed ONCE, by its first KMH run (`kmh_confirm/nq50/kmh_reference`, the first job of the plan),
and loaded by the other four KMH groups of the seed (raw and `l2`, all query sizes). A few minutes per seed, not
added to the estimate.

### Seed audit across the two accounts

The initial labeled set of a run is drawn by `Data.initialize_labels` (`np.random.shuffle` of the pool indices)
right after `seed_everything(seed)` in `ExperimentRunner.run`; nothing between them draws from NumPy's legacy
stream (the executed seed audits PASS across all strategies, including KMH and WAAL, and across the two accounts),
and it does not depend on `n_query` or the strategy. So every seed-4 run of `kmh_confirm` (account A) starts from
the same initial images as every seed-4 run of `seeds_ext` (account B), and likewise for seeds 5 and 6.

`audit_seed_consistency` (unchanged) groups the selected jobs by seed only, whatever their part or manifest, and
compares the round-0 image paths (`round_queries.json`) of each run with the first readable run of that seed:

- on account A, `make campaign-verify-gpu-part GPU_PART=part_1_gpu` compares the 27 `kmh_confirm` runs of each of
  seeds 4, 5, 6 among themselves (the only seed 4-6 jobs of `part_1_gpu`);
- after merging both zips locally, `make campaign-verify-all` (both manifests, `--part all`) puts the `kmh_confirm`
  and `seeds_ext` runs of a seed in one comparison (`FullSupervised` excluded), so a difference between the
  accounts is a FAIL. Tested: `tests/test_campaign.py::test_seed_audit_pools_kmh_confirm_and_seeds_ext_by_seed`;
  micro evidence in `.specs/experiments/campaign.md`.

Post-run check (as in ADR 0019 for round 0): within each family the round-0 weighted F1 of a seed must be equal
across query sizes and accounts (KMH raw and `l2`: every `kmh_confirm` KMH group and `seeds_ext/KMH`; RandomSampling:
`kmh_confirm/nq{50,10}/RandomSampling` and `seeds_ext/RandomSampling`). On seeds 1-3 this held for every executed
pair (e.g. RandomSampling seed 1: 0.2406 at nq100 on account A, at nq50 and nq10 on account B; KMH seed 1: 0.3839).
KI-50 is kept on purpose, as in ADR 0019, so the KMH rows start from the round-0 model of every executed KMH run of
the same seed.

### GPU time estimates

`basis: estimate`, rule `scripts/campaign/build_gpu_time_estimates.py::confirm_estimate`: the counterpart's mean
wall time per run (`DONE elapsed=<s>` of its three seeds) MEASURED ON THE CLEAN CAMPAIGN
(`results/campaign_fixed/campaign_part_2_gpu.log` for the paper-1 counterparts, account B;
`results/campaign_kmh_fixed/campaign_part_1_gpu__kmh_norm.log` for `kmh_norm` l2, account A), written as constants
because those logs exist on the author's machine only. The first-execution values are not used: the clean campaign
measured WAAL at 3,474-3,484 s (nq100), 3,571-3,598 s (nq50) and 3,647-3,651 s (nq10) per run, against the 2,700 s
FLOP estimate of the table. Total: 30,546 s = **8.5 GPU h** for the 27 runs (WAAL 7,239 s per seed, 6.0 h of
them). The paper-1 counterparts ran on account B's GPU; account A's may be another model (`run_metadata.json`).

## What the confirmation can and cannot show

- **Can**: whether `l2`'s gain over raw features at nq100 and nq50 (and its loss at nq10) reappears on three seeds
  that played no part in choosing it, with the same initial labeled images per seed as the `seeds_ext` headline
  runs, and where `l2` stands against WAAL and RandomSampling on those seeds. The seeds-4-6 numbers of `l2` are free
  of the selection effect of choosing it among three variants on seeds 1-3.
- **Cannot**: create a validation split (every number is still a test-set number); settle a small difference (three
  seeds, with seed spreads of 5 to 10 points at nq50/nq10); say anything about `zscore`, `pca64_l2`, other `pca_dim`
  values or other hierarchies (none of them runs on seeds 4-6); change the seeds-1-3 tables.
- **No threshold is set here**: the report gives, per query size, the seeds-4-6 mean and standard deviation of the
  four rows, and the per-seed paired difference `l2` minus reference is readable from the leaves (same seed = same
  initial images and same round-0 model). The claim wording is the author's decision (pendencia P42) and must name
  the seeds-4-6 tables.

## Consequences

- Main manifest 384 runs / 128 groups (357 / 119 + 27 / 9); `part_1_gpu` 363 / 121 (336 / 112 + 27 / 9);
  `part_2_gpu` unchanged 252 / 84; both manifests 615 runs / 205 groups; estimated total 128.1 GPU h. The KMH sweep
  manifests are byte-identical to v5 (and v6); the 196 groups of tag `campaign-fixed-v6` (588 runs) keep id, parts,
  stage, params, budgets, seeds, results dir and effective signature (pinned v6 digests computed with the v6 code
  on a checkout of the tag, `test_v6_groups_keep_their_identity`). In the main manifest file only the `used_by` and
  `note` of `seeds_ext/{KMH,RandomSampling,WAAL}` change (the new alias rows), neither is part of a group's identity.
- Engine: `git diff --stat campaign-fixed-v6 -- dalmax/ tools/ pyproject.toml poetry.lock` shows only the part list
  of `dalmax/campaign.py` (`PARTS`, `PHASE_LABELS`, docstring). `tools/trainer.py` does not import `dalmax.campaign`.
- Pilot: `part_1_gpu`'s pilot stays `shared/random_nq100`, `p1/VAAL/nq100`, `p1/WAAL/nq100`, seed 1 (the pilot is
  picked only among groups that run seed 1, ADR 0020): all three executed, all SKIP.
- Tag **`campaign-fixed-v7`** (created by the author after review) = v6 + this part, config-only relative to v6, so
  the trainer code of v7 is that of v5 and v6. Only account A uses v7: its code-version audit then lists four
  commits (v2 for the 285 earlier leaves, v3 for the 24 `texhal_q13` leaves, v5 for the 27 `kmh_norm` leaves, v7 for
  the 27 new ones), an explained WARN; after merging both accounts locally up to six commits (v2 to v7).
- Account B stays on v6 until `descriptors_nq` + `seeds_ext` are merged (no second code version inside a part).
  Opening the v7 notebook on account B by mistake would change no computation (its plan and `EXPECTED` (252, 84) are
  the same under v7, and the trainer code is identical), but every leaf it writes would record the v7 commit: a
  fourth commit on that account, forbidden by the one-tag-per-part rule, and not caught by the banner (same counts).
  `COLAB_RUNBOOK.md` section 6i says so.
- Notebook (`scripts/colab/build_colab_notebook.py`): `DEFAULT_GIT_REF = "campaign-fixed-v7"`, `EXPECTED`
  `part_1_gpu` (363, 121), `part_2_gpu` (252, 84); `ZIP_ONLY_PARTS` gains `kmh_confirm`
  (`ZIP_ONLY_PART = "kmh_confirm"`: `part_1_gpu__kmh_confirm.zip`, 27 checkpoints). Runbook: `COLAB_RUNBOOK.md`
  section 6i.
- Local verify: today (account B's 36 runs and this part not merged) `make campaign-verify-all` expects OK=552,
  MISSING=63 (36 + 27); OK=615 after both merges.
