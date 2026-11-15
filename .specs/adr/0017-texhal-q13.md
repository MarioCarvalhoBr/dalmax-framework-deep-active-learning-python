# ADR 0017: TexHAL canonical configuration Q=13 on the clean dataset; run its 6.2/6.3 ablations (part `texhal_q13`)

- **Status:** Accepted (author decision 2026-11-15); supersedes the `Q=17` choice of ADR-0011; prepared, not executed
- **Date:** 2026-11-15

## Context

ADR 0011 made VCTex `Q=17` (single RAE) TexHAL's canonical configuration from the first campaign, which ran on
`DATA/daninhas_full`. That evidence is unreliable for two reasons found later
(`.specs/quality/known-issues.md`):

- **KI-46**: `DATA/daninhas_full` holds byte-identical duplicates, 171 images of them in both train and test, so
  every number of the first campaign was measured on a test set that leaks into the pool.
- **KI-47**: the loaders listed each class folder with an unsorted `os.listdir`, whose order differed between
  Colab sessions, so the same seed started from other images in later sessions and the Drive-resident embedding
  cache (keyed on pool indices) served features of other images; `texhal_q17` (the 6.2/6.3 re-run of ADR 0011)
  was among the affected parts.

The clean campaign (ADR 0016; 462 runs on `DATA/daninhas_full_fixed`, tag `campaign-fixed-v2` = commit
`464a5cc`; executed, merged locally and verified: `make campaign-verify-all` OK=462, one commit, seed and
data-order audits PASS, the three KMH duplicate pairs SAME) re-ran the 6.1 representation table of paper 2. At
`n_query=100` (final-round weighted F1, mean of seeds 1-3, std in parentheses):

| VCTex Q | weighted F1 |
|---|---|
| `Q=13` (`texhal/6_1/rep_q13`) | **0.8341** (0.0188) |
| `Q=[5,17]` (`shared/texhal_full`) | 0.8038 (0.0173) |
| `Q=5` (`texhal/6_1/rep_q5`) | 0.7766 (0.0639) |
| `Q=17` (`texhal/6_1/rep_q17`, ADR 0011's choice) | 0.7703 (0.0519) |
| KMH (ImageNet ResNet-50 features, reference hierarchy, `shared/kmh_nq100`) | 0.7921 (0.0617) |

The TexHAL hierarchy (6.2) and stage (6.3) ablations exist only for `Q=17` (`texhal_q17`, plus its two
`hier_transfer` rows) and `Q=[5,17]` (`texhal`).

## Decision

We will make **VCTex `Q=13` (single RAE, reference hierarchy L=3 `k=[600,200,100]`)** TexHAL's canonical
configuration for paper 2, paper 3's comparison and the thesis, replacing ADR 0011's `Q=17`, and we will execute
its 6.2 and 6.3 ablations as a new campaign part **`texhal_q13`** that mirrors `texhal_q17` exactly, with only the
embedding `q` changed:

- 8 run groups / 24 runs at `n_query=100`, seeds 1-3: 6.2 `hier_L1`, `hier_L2a`, `hier_L2b`, `hier_L3`, `hier_L4`,
  `hier_L2_matched__no_resampling`, `hier_L3__no_resampling`; 6.3 `stage_no_hierarchy`. Params in
  `files_config/ablations/texhal_q13/` (+ `micro/`), each equal to its `texhal_q17` twin except `embedding.q = [13]`.
- No new run for the other rows: `texhal_q13/6_2/hier_ref` and `texhal_q13/6_3/stage_full` are aliases of the
  executed `texhal/6_1/rep_q13` (its config equals `texhal/stage_full.json` with `q=[13]`, the same justification
  as the `texhal_q17` aliases of `rep_q17`), and `texhal_q13/6_3/stage_no_representation` is `shared/kmh_nq100`.
- Results under `results/campaign_fixed/texhal_q13/`, GPU part `part_1_gpu` (account A: its Drive already holds
  the VCTex Q=13 embedding cache written by `texhal/6_1/rep_q13` under the same data order), GPU stage 2
  (ablations). GPU time estimates are the measured `texhal_q17` twins, marked `estimate`
  (`scripts/campaign/build_gpu_time_estimates.py::TWIN_PREFIXES`), about 1.4 GPU h in total.
- Reports (`scripts/campaign/build_manifest.py`): paper 2's tables `6_2`/`6_3` (ids unchanged) come from
  `texhal_q13`; the executed Q=17 tables stay as the supplementary `6_2_q17`/`6_3_q17`, next to the history
  tables `6_2_q5_17`/`6_3_q5_17`; the TexHAL row of the comparison tables of papers 2 and 3 becomes
  "TexHAL (VCTex Q=13 + hierarchical)" -> `texhal/6_1/rep_q13`; in 6.1 the "(TexHAL)" tag moves from `Q=17` to
  `Q=13`. Nothing executed is dropped from the reports.

## Disclosed limitations

- **Q is chosen on the test set.** The four Q values are compared by their final-round test F1, and the winner is
  then reported on the same test set; there is no separate validation split. The author knows and accepts this;
  the thesis discloses it, and the paper-2 text must not present `Q=13` as selected independently of the test set.
- `Q=13` is also SSRAE's hidden-layer size (RNHAL, paper 3); the two methods remain different descriptors
  (VCTex texture RAE vs SSRAE spatio-spectral RAE), but the coincidence should be stated, not left to be noticed.
- Three seeds only: the `Q=13` vs `Q=17` gap (0.8341 vs 0.7703, 6.4 points) is about 1.2 std of `Q=17` (0.0519),
  and `Q=5`/`Q=17`/KMH overlap within one std. The ranking of the four Q values also differs from the first
  campaign's (`Q=17` best there, on the KI-46/KI-47-affected runs).

## Consequences

- Main manifest 321 runs / 107 groups (was 297 / 99); `part_1_gpu` 309 runs / 103 groups (was 285 / 95);
  `part_2_gpu` unchanged (177 / 59); both manifests 486 runs (was 462). The KMH sweep manifest is byte-identical.
  The 99 executed groups of the main manifest and the 55 of the sweep keep id, params file, results directory,
  parts, stage and effective signature (`tests/test_campaign.py::test_executed_groups_keep_their_identity` pins a
  digest of them); only `used_by`/`note` metadata changed (the comparison tables now consume `rep_q13`, not
  `rep_q17`).
- `dalmax/campaign.py` gains the semantic part name `texhal_q13` (`PARTS`, `PHASE_LABELS`, after
  `hier_transfer`); nothing a `tools/trainer.py` run executes changes, so the run-affecting code of the 24 new runs
  is the code of the 462 executed ones.
- New tag **`campaign-fixed-v3`** (created by the author after review) is the notebook's `DEFAULT_GIT_REF`. On
  account A the code-version audit of `campaign verify` then lists **two commits** (`464a5cc` for the 285 earlier
  leaves of `part_1_gpu`, the v3 commit for the 24 new ones), a `WARN` that stays a WARN by design and is expected
  and explained here and in `COLAB_RUNBOOK.md` section 6e; any third commit, or a missing one, still needs an
  explanation before the results are used.
- The notebook's zip cell gains `ZIP_ONLY_PART` (default `""` = unchanged behaviour); `"texhal_q13"` ships only
  `results/campaign_fixed/texhal_q13/` (24 checkpoints) plus the renamed logs, merged locally with `unzip -n`.
- ADR 0011 is superseded as to the canonical Q; its executed runs (first campaign) and the clean `texhal_q17`
  runs stay as they are (append-only) and are reported as supplementary.
- Defaults in `dalmax/config/loader.py` (`vctex` -> `(5, 17)`) and the VCTex presets are not changed (no code
  change; TexHAL is always configured through its params JSON).
- Follow-up after execution: record the 6.2/6.3 outcome here and in `.specs/experiments/campaign.md`; the thesis
  analysis (`thesis_drafts/`) must follow the relabelled rows (6.1 `Q=13 (TexHAL)` / `Q=17`, comparison label
  "TexHAL (VCTex Q=13 + hierarchical)") and the new table ids `6_2_q17` / `6_3_q17`.
