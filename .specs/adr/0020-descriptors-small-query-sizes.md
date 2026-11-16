# ADR 0020: Run the descriptor methods at query sizes 50 and 10, and the headline comparison on seeds 4-6 (parts `descriptors_nq` and `seeds_ext`, account B)

- **Status:** Accepted (author decision 2026-11-16, thesis pendencia P36 and the three-seed review criticism);
  prepared, not executed
- **Date:** 2026-11-16

## Context

The clean campaign (ADR 0016, `DATA/daninhas_full_fixed`) is executed and verified locally for 525 runs (OK=525;
`kmh_norm`, 27 runs, is running on account A with tag `campaign-fixed-v5`, ADR 0019). Final-round weighted F1, mean
of seeds 1-3:

- The two descriptor methods exist only at `n_query = 100`: RNHAL (`rnhal/6_3/stage_full`, SSRAE `Q=13`, variant
  `full`, reference hierarchy `k=[600,200,100]`, `sample_sizes=[30,15,2]`) 0.8608, and TexHAL (`texhal/6_1/rep_q13`,
  VCTex `q=[13]`, same hierarchy; canonical since ADR 0017) 0.8341. Both end above the best benchmark strategy, WAAL
  (0.8077), in each of the three seeds.
- The benchmark (paper 1) also has `n_query` 50 and 10: WAAL 0.7874 and 0.5793, RandomSampling 0.6455 and 0.4756.
  Whether the descriptor methods keep their advantage at the smaller query sizes is unknown (pendencia P36).
- Two independent reviews of the drafts criticised the use of three seeds. At `n_query = 100` the ranking of RNHAL
  and TexHAL against KMH and BALD is not consistent across seeds 1-3 (the margins change sign or size from seed to
  seed), so three seeds cannot settle those comparisons.

Colab account B (`part_2_gpu`) is idle: its part is executed (paper 1 at nq50/nq10, the nq10 KMH sweep, `kmh_wide`).

## Decision 1: part `descriptors_nq` (RNHAL and TexHAL at `n_query` 50 and 10)

We will run RNHAL and TexHAL in their executed `n_query = 100` configuration at `n_query` 50 and 10, as a new part
**`descriptors_nq`** of the main manifest (`files_config/campaign/manifest.json`, generator
`scripts/campaign/build_manifest.py`, micro mirror `manifest_micro.json`): 4 run groups x seeds 1-3 = 12 runs,
`RepresentationStrategy`, NIL 100, 8 rounds.

| group id | params file (reused) | `n_query` | executed nq100 counterpart | estimate (s/run) |
|---|---|---|---|---|
| `descriptors_nq/rnhal/nq50` | `files_config/ablations/rnhal/stage_full.json` | 50 | `rnhal/6_3/stage_full` | 556 |
| `descriptors_nq/rnhal/nq10` | `files_config/ablations/rnhal/stage_full.json` | 10 | `rnhal/6_3/stage_full` | 541 |
| `descriptors_nq/texhal_q13/nq50` | `files_config/ablations/texhal/rep_q13.json` | 50 | `texhal/6_1/rep_q13` | 599 |
| `descriptors_nq/texhal_q13/nq10` | `files_config/ablations/texhal/rep_q13.json` | 10 | `texhal/6_1/rep_q13` | 582 |

- **Params**: each group points to the params file of its executed nq100 run (the query size is a manifest/CLI
  value, not in the params), the way `p1/KMH/nq50` and `p1/KMH/nq10` reuse `params_kmh.json`. No new params file.
  Test: the effective signature of each group equals its counterpart's except `n_query`
  (`tests/test_campaign.py::test_descriptors_nq_part_is_the_executed_nq100_configuration_at_nq50_and_nq10`).
- **Layout**: results under `results/campaign_fixed/descriptors_nq/<method>/nq{N}/` (`<method>` = `rnhal` or
  `texhal_q13`), a folder of its own; `gpu_part = "part_2_gpu"` (account B), `gpu_stage` 2 (the ablations stage).
  The groups are appended after every group of tag `campaign-fixed-v5`, method-major (the nq50 run of a seed is
  planned before its nq10 run; the cache key, below, ignores `n_query`).
- **Feasibility**: the reference hierarchy at `n_query` 10 and 50 is already executed with ImageNet features
  (`p1/KMH/nq10`, `p1/KMH/nq50`); only the features differ here. Confirmed by the micro run (execution record in
  `.specs/experiments/campaign.md`).
- **Embedding cache**: account B's Drive has no SSRAE or VCTex `Q=13` cache (both were written on account A). The key
  (`dalmax/embeddings/cache.py`) is `(dataset, extractor, Q, variant, split, order_hash, pool_hash)`, with
  `pool_hash` over the INITIAL unlabeled pool, fixed by the seed and `n_init_labeled` only: the nq50 run of a seed
  computes the cache, the nq10 run of the same seed loads it. Extra time, measured on the thesis side: about 83 s
  per seed for SSRAE and 43 s per seed for VCTex (about 6 minutes for the part).
- **GPU time estimates** (`basis: estimate`, rule `scripts/campaign/build_gpu_time_estimates.py::scaled_estimate`):
  the measured seconds of the method's nq100 run times the ratio of the measured paper-1 KMH runs
  `p1/KMH/nq{N}` / `shared/kmh_nq100` (0.968 at nq50, 0.941 at nq10). KMH is the same hierarchy and training on
  other features, so the ratio carries the cost of the query size and nothing else. Total 1.9 GPU h.
- **Report**: two new paper-1 tables, `descriptors_nq50` and `descriptors_nq10`: RNHAL, TexHAL, then the executed
  paper-1 runs of the same query size, WAAL, BALDDropout, BadgeSampling, KMH (`p1/KMH/nq{N}`) and RandomSampling.
  They live in paper 1's report so that every paper-1 group stays consumed by one paper only (the `shared/`
  ownership rule, ADR 0009); existing table ids and rows are unchanged.

### What `descriptors_nq` can and cannot show

- The configuration is fixed beforehand (the executed nq100 one): nothing is selected on the test set by this round.
- The reference hierarchy was designed for `n_query = 100` (its last level has exactly 100 clusters). A weak result
  at `n_query = 10` would not show that the descriptors fail at small query sizes, only that this hierarchy does
  (the KMH sweep, ADR 0012/0018, shows how much the shape matters at nq10 with ImageNet features). A strong result
  would show that the gain survives without retuning.
- Three seeds, with the large seed spread of the small query sizes (5 to 10 points at nq50/nq10 in the KMH sweep):
  a small difference is not a finding.

## Decision 2: part `seeds_ext` (the headline comparison on seeds 4, 5 and 6)

We will re-run the headline comparison of papers 2 and 3 at `n_query = 100` on three NEW seeds, 4, 5 and 6, as a
new part **`seeds_ext`** of the main manifest: 8 run groups x seeds `[4, 5, 6]` = 24 runs. Each group is an exact
copy of its executed seeds-1-3 counterpart (same params file, strategy, `n_query`, `n_init_labeled`, `n_round` and
`gpu_stage`; the same effective signature, tested), only the seeds and the results folder differ:

| group id | counterpart (seeds 1-3) | strategy | `gpu_stage` | estimate (s/run) |
|---|---|---|---|---|
| `seeds_ext/RandomSampling` | `shared/random_nq100` | RandomSampling | 1 | 72 |
| `seeds_ext/BALDDropout` | `p1/BALDDropout/nq100` | BALDDropout (MC-dropout params) | 1 | 239 |
| `seeds_ext/BadgeSampling` | `p1/BadgeSampling/nq100` | BadgeSampling | 1 | 92 |
| `seeds_ext/KMH` | `shared/kmh_nq100` | RepresentationStrategy (`params_kmh.json`) | 1 | 590 |
| `seeds_ext/upper_bound` | `p1/upper_bound` | FullSupervised (NIL 7,230, NR 0) | 1 | 51 |
| `seeds_ext/rnhal` | `rnhal/6_3/stage_full` | RepresentationStrategy | 2 | 575 |
| `seeds_ext/texhal_q13` | `texhal/6_1/rep_q13` | RepresentationStrategy | 2 | 619 |
| `seeds_ext/WAAL` | `p1/WAAL/nq100` | WAAL | 5 | 2,700 |

- **Layout**: results under `results/campaign_fixed/seeds_ext/<method>/` (leaves
  `.../daninhas_full_fixed/SEED_{4,5,6}/NQ_..._NIL_..._NR_..._NE_.../<strategy>/`); `gpu_part = "part_2_gpu"`;
  appended after `descriptors_nq`. Seeds 4-6 are never merged into a seeds-1-3 group or table.
- **Report**: one new paper-3 table, `seeds_ext_nq100`, listing the 8 groups (each row aggregates its own group's
  seeds 4-6). No existing table id, row or number changes; no `best_of` candidate points to a `seeds_ext` group
  (tested).
- **Estimates** (`basis: estimate`, rule `build_gpu_time_estimates.py::counterpart_estimate`): the counterpart's
  entry (measured, or the counterpart's own estimate for WAAL and BadgeSampling). Total 4.1 GPU h; with
  `descriptors_nq`, 6.0 GPU h for the 36 runs. The SSRAE, VCTex `Q=13` and ImageNet caches of seeds 4-6 exist
  nowhere: each is computed by the first run of its extractor and seed (SSRAE about 83 s and VCTex about 43 s per
  seed, thesis-side measurement; the ImageNet ResNet-50 forward pass over the pool, not measured separately, a few
  minutes at most per seed on a GPU), not added to the estimates.
- **KI-50** (the ImageNet provider draws from the global torch stream at construction) applies to `seeds_ext/KMH`
  exactly as to `shared/kmh_nq100`: its round-0 classifier starts from another point of the stream than the other
  strategies of the same seed. Not fixed here: the KMH row on seeds 4-6 stays the same computation as on seeds 1-3.

### Engine changes (campaign engine only; nothing `tools/trainer.py` executes)

`git diff --stat campaign-fixed-v5 -- dalmax/ tools/ pyproject.toml poetry.lock` shows `dalmax/campaign.py` only:

1. `PARTS` / `PHASE_LABELS` / docstring gain `descriptors_nq` and `seeds_ext` (after `texhal_q13`, before
   `paper1_other_nq`), as for `texhal_q13` (ADR 0017).
2. `find_redundant_groups` (the "nothing runs twice" check of `validate_manifest`, ADR 0009) flags two groups only
   when they are the same computation **on a common seed**. Before, it ignored seeds and would have rejected every
   `seeds_ext` group as a duplicate of its counterpart. Every group of tag v5 runs seeds 1-3, so no v5 decision
   changes (tested: `test_redundancy_is_the_same_computation_on_a_common_seed`, and the manifests still validate).
3. `pilot_group_ids` considers only the groups that run the pilot seed (the smallest seed of the selection, 1).
   Without it, `seeds_ext/WAAL` (nq100) would have become the largest-budget WAAL of `part_2_gpu`, and the pilot
   would have failed for lack of a VAAL group at nq100. With it, the pilot of `part_2_gpu` stays
   `p1/RandomSampling/nq50`, `p1/VAAL/nq50`, `p1/WAAL/nq50`, seed 1: the three executed pilot leaves, all SKIP.

Unchanged and checked: `cross_manifest_duplicates` keeps the first group of a signature it meets
(`shared/kmh_nq100` before `seeds_ext/KMH`), so the KMH sweep's reference still pairs with the same three groups and
`check_duplicate_pairs` only compares common seeds (still 3 intentional pairs). `expand_jobs` lists a group's own
seeds, so `SEEDS=1` (smoke) gives no `seeds_ext` job and `SEEDS=4` only `seeds_ext` jobs. The report aggregates
each group over its own seeds (`campaign_report.load_group_runs`).

**Seed audit for seeds 4-6** (`audit_seed_consistency`): unchanged code. It groups the runs of a selection by seed
and compares each run's round-0 image paths (`round_queries.json`) with the first readable run of that seed in plan
order, `FullSupervised` excluded. For seeds 4, 5 and 6 the runs compared are the 7 non-`FullSupervised` groups of
`seeds_ext`, and the reference is the first of them in plan order (`seeds_ext/RandomSampling` in `part_2_gpu`); a
seed with no leaf yet is `NO_DATA`. So the audit is meaningful for the new seeds without a change: every run of
seed 4 must start from the same initial labeled images, and so on. Evidence from the micro run in
`.specs/experiments/campaign.md`.

### What `seeds_ext` can and cannot show

- Seeds 4-6 were not looked at before (no run exists): no configuration or method is selected on them. The 8 methods
  and their configurations are fixed by the seeds-1-3 campaign.
- Six seeds instead of three narrow the seed-to-seed uncertainty of the headline comparison; they do not create a
  validation split (every number is still a test-set number), and they do not touch the ablations, which stay at
  three seeds.
- How the papers report it is decided by the author when the results arrive. Planned: six seeds (1-6) for the
  headline comparison of papers 2 and 3, three seeds for the ablations. The six-seed numbers are computed from the
  two tables (`comparison` / paper-1 rows on seeds 1-3, `seeds_ext_nq100` on seeds 4-6); the report keeps them
  apart.
- WAAL's seeds-4-6 run is the most expensive item (45 minutes per run, estimate) and runs last (stage 5).

## Consequences

- Main manifest 357 runs / 119 groups (321 / 107 + 12 / 4 + 24 / 8); `part_2_gpu` 252 / 84 (216 / 72 + 36 / 12);
  `part_1_gpu` unchanged 336 / 112; both manifests 588 runs. The KMH sweep manifests are byte-identical to v5; the
  184 groups of v5 (552 runs) keep id, parts, stage, params, budgets, seeds, results dir and effective signature
  (pinned v5 digests computed with the v5 code on a checkout of the tag, `test_v5_groups_keep_their_identity`).
- Tag **`campaign-fixed-v6`** (created by the author after review) = v5 + these two parts, config-only relative to
  v5 (the engine changes above are not run by a trainer subprocess). Only account B uses v6: its audit then lists
  three commits (v2 for the 177 earlier leaves, v4 for the 39 `kmh_wide` leaves, v6 for the 36 new ones), an
  explained WARN; after merging both accounts locally up to five commits (v2, v3, v4, v5, v6). Account A stays on
  v5 until `kmh_norm` is merged (no second code version inside a part).
- Notebook (`scripts/colab/build_colab_notebook.py`): `DEFAULT_GIT_REF = "campaign-fixed-v6"`, `EXPECTED`
  `part_2_gpu` (252, 84); `ZIP_ONLY_PART` accepts several parts of the notebook's GPU part joined by `+`
  (`"descriptors_nq+seeds_ext"`: one zip `part_2_gpu__descriptors_nq__seeds_ext.zip` with 36 checkpoints, logs of
  the main root renamed `campaign_part_2_gpu__descriptors_nq__seeds_ext.log` / `failures_...`; the names are joined
  by `__` in every file name so that no `+` reaches the shell or the Drive); the single-part values keep working.
  Runbook: `COLAB_RUNBOOK.md` section 6h.
- Local verify after merging account B's zip: `make campaign-verify-all` expects OK=561 and MISSING=27 while
  `kmh_norm` is not merged (OK=588 after it).
- The seeds-1-3 numbers of every paper are unchanged by this ADR; any six-seed statement in the thesis is new text
  and must name both tables.
