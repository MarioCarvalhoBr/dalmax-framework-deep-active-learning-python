# ADR 0016: Re-run every experiment of the three papers on the duplicate-free dataset, in two GPU parts

- **Status:** Accepted (author decisions 2026-11-11); prepared 2026-11-11, not executed. Supersedes the operational
  part of ADR 0014 (re-run parts `paper1_rerun_a`/`paper1_rerun_b`, superseded-leaf guard, archive step); ADR 0014's
  code fixes (KI-41, KI-42, KI-45) stand.
- **Date:** 2026-11-11

## Context

- **KI-46** (`.specs/quality/known-issues.md`): `DATA/daninhas_full` holds byte-identical copies of the same image,
  inside each split and across train and test (train 8,086 files / 7,405 distinct, test 2,106 / 1,008 distinct, 171
  distinct images in both splits = 394 test files and 198 train files, never across classes). Every result of the
  first campaign (`results/campaign/`, `results/campaign_kmh/`) was measured on a test set where about 19% of the files
  are also in the pool and many test images count several times. The author decided (2026-11-11) to re-run
  **everything** on a new canonical dataset, `DATA/daninhas_full_fixed` (7,230 train / 1,008 test images plus
  `MANIFEST.csv` and `README.txt`, 8,240 files; built by `scripts/datasets/build_daninhas_full_fixed.py`, commit
  c7d2e9a; `--check` verifies that no MD5 repeats), reusing no old result.
- ADR 0015 added eight DeepAL+ strategies (`VarRatio`, `MeanSTD`, `KCenterGreedyPCA`, `BadgeSampling`,
  `CEALSampling`, `LossPredictionLoss`, `VAAL`, `WAAL`) that were in no manifest; the author wants them in paper 1 at
  the three budgets, like the 12 classical strategies.
- ADR 0014 planned to re-run 21 fixed groups into the SAME leaf paths as their first execution, which required a
  guard (`dalmax/superseded.py`, run refusal, `STALE` verify status) and an archive move under `results/`. With
  everything re-run into new roots that hazard no longer exists.
- The work has to run on two Google Colab accounts in parallel. The first execution measured about 1.7 points of
  weighted-F1 variation between runtime sessions (KI-43), so results compared in one table should come from one
  session where possible.
- When KI-46 was found, per-round metrics could not be recomputed on a de-duplicated test set: only the last round's
  `predictions.csv` was saved.

## Decision

We will:

1. **Repoint** every full-scale params JSON of the campaign, the KMH sweep, the ablations and the lab/Colab checks to
   `DATA/daninhas_full_fixed/` (generators updated and re-run where they exist: `build_kmh_sweep_configs.py`; the
   hand-written ablation and campaign files edited). Micro configs keep `DATA/daninhas_micro/` (the golden fixtures
   depend on it). The embedding cache and the results leaf already key on the dataset folder name, so nothing
   collides with the old dataset's caches or leaves.
2. **Write to fresh roots**: `results/campaign_fixed/` (main manifest, `manifest.json`), `results/campaign_kmh_fixed/`
   (KMH sweep, `manifest_kmh_sweep.json`), reports under `docs/results/campaign_fixed/` and
   `docs/results/campaign_kmh_fixed/` (the report's default `--out` is `docs/results/<last component of the root>`);
   micro smoke under `results/smoke_campaign_fixed/` and `results/smoke_campaign_kmh_fixed/`; the KI-43 control under
   `results/campaign_control_fixed/`. The old trees are never touched. The manifests of the first execution remain in
   git history (commit d668673 and earlier).
3. **Retire the ADR 0014 machinery**: the 21 groups return to part `paper1`; `dalmax/superseded.py`,
   `scripts/campaign/archive_superseded_runs.py`, `make campaign-archive-superseded`, the run guard (exit 3) and the
   `STALE` verify status are deleted, with their tests (`tests/test_campaign_superseded.py`,
   `tests/test_archive_superseded_runs.py`). Rationale: in fresh roots a superseded leaf cannot exist, so the guard
   would only be dead code that suggests a hazard; keeping a script that moves directories under `results/` without a
   use would keep an exception to the append-only rule alive for nothing. The executed first-campaign leaves that ADR
   0014 would have archived stay where they are (`results/campaign/paper1/`), as history of the first campaign.
4. **Main manifest**: paper 1 = the 20 strategies (12 classical + 8 DeepAL+) + KMH at `n_query` 10/50/100 + the upper
   bound (`n_init_labeled` = 7,230); papers 2/3 unchanged in content (`rnhal`, `texhal`, `texhal_q17`,
   `hier_transfer`). Params: the dropout family (the four MC-dropout strategies, `VarRatio`, `MeanSTD`) uses
   `params_paper1_mc_dropout.json`; `VAAL` uses `params_paper1_vaal.json` with `image_size: 32` (author's cost
   decision; the VAE takes any multiple of 16, covered by `tests/test_deepalplus_strategies.py`); every other strategy
   uses `params_paper1.json`. Totals: 294 runs / 98 groups (paper1 189 / 63). The KMH sweep keeps 165 runs / 55 groups.
5. **GPU parts**: a second partition, orthogonal to the semantic parts. Every group of both full-scale manifests has
   `gpu_part` in {`part_1_gpu`, `part_2_gpu`} and a `gpu_stage` (execution stage inside the part). `--part part_1_gpu`
   (or `part_2_gpu`) selects by it in `campaign list|run|verify`; GPU parts and semantic parts cannot be mixed in one
   `--part`; `all` still means every group. `--manifest` accepts a comma-separated list whose jobs are merged into one
   plan (by stage for a GPU part), so `make campaign-run-gpu-part GPU_PART=...` runs both manifests for one account,
   resumable by re-running; `make campaign-list-gpu-part` / `campaign-verify-gpu-part` match it.
   - Assignment (`scripts/campaign/build_manifest.py`): `part_1_gpu` = paper 1 at nq100 (with `shared/random_nq100`,
     `shared/kmh_nq100`), the upper bound, all of `rnhal`, `texhal`, `texhal_q17`, `hier_transfer`, and the KMH sweep at
     nq100 and nq50; `part_2_gpu` = paper 1 at nq50 and nq10 and the KMH sweep at nq10. A group never splits.
   - Balance: estimated 47.5 GPU h (`part_1_gpu`, 282 runs / 94 groups) vs 48.6 h (`part_2_gpu`, 177 runs / 59
     groups), from `files_config/campaign/gpu_time_estimates.json` (see below).
   - Stages: 1 paper-1 strategies other than the long ones, KMH and the upper bound; 2 the ablations; 3 the KMH sweep;
     4 `AdversarialBIM`; 5 `VAAL`, `WAAL`; 6 `AdversarialDeepFool`. An exhausted credit budget therefore leaves complete
     tables except the long baselines' rows.
   - Comparisons that cross the two parts: paper 1 at nq50 and nq10 against the upper bound (`part_1_gpu`); paper 1 at
     nq50 against the KMH sweep at nq50 (paper 1 reports KMH per budget from the sweep, ADR 0012); any cross-budget
     reading of paper 1 (nq100 in part 1, nq50/nq10 in part 2). Every other table (paper 1 per budget, papers 2/3
     6.1/6.2/6.3 and comparisons, every KMH-sweep table) is inside one part.
6. **GPU-time estimates as data**: `files_config/campaign/gpu_time_estimates.json`, built by
   `scripts/campaign/build_gpu_time_estimates.py` from the first execution's trees (read only): per group the mean
   trainer-subprocess wall time per run (`campaign.log` `DONE elapsed`, or `Total time` + the median 11 s overhead for
   runs merged from other sessions); for the eight new strategies, rules from the author's brief, marked `estimate`
   (`LossPredictionLoss`, `BadgeSampling`, `KCenterGreedyPCA` about a plain uncertainty strategy, `VarRatio`/`MeanSTD`
   about an MC-dropout one, `CEALSampling` 1.25 times plain, `WAAL` about 45 min, `VAAL` at 32 px about 50 min per run).
   The manifests reference it (`gpu_time_estimates`) and `campaign list` prints estimated hours per part, per GPU part
   and in total. No GPU model is named.
7. **Signature**: `dalmax.campaign.effective_signature` reduces `strategy_args` to the block the strategy reads
   (`STRATEGY_ARGS_KEYS`; none for the others), so params files differing only in another strategy's block are the same
   computation (e.g. `RandomSampling` with `params_paper1_vaal.json` equals `params_paper1.json`). No dedup decision of
   an executed group changes: none of their params had a `strategy_args` block.
8. **Per-round history**: every run additionally writes `round_predictions.json` (the predicted class of every test
   image at every round, with test paths and true classes) and `round_queries.json` (per round the sorted pool
   indices labeled in it, round 0 = the initial set, with true classes and image paths; the pool order follows
   `os.listdir`, so the paths are the stable key). `results.json`, `predictions.csv` and the log lines are unchanged.
   `dalmax/reporting/leaf_check.py::check_leaf(leaf, extra_required)` checks them when a manifest lists them in
   `required_leaf_files` (all manifests of this campaign do); leaves of the old trees still verify without them.
9. **Colab hand-off**: `scripts/colab/setup_colab.sh` uses `daninhas_full_fixed` (zip `daninhas_full_fixed.zip`,
   8,240 files checked against the real folder) and runs the builder's `--check` after unzipping; `DRIVE_ROOT` has no
   personal default any more (required). `notebooks/colab_runbook.ipynb` is now generated by
   `scripts/colab/build_colab_notebook.py` (drift-tested) with a single `GPU_PART` parameter and gate cells that raise
   on a wrong value, an unfilled `DRIVE_ROOT`, a missing/wrong dataset, a dirty checkout, leaves of the other GPU part
   on the Drive, or a job count different from the manifests. `COLAB_RUNBOOK.md` section 6 is the operator guide.

## Consequences

- Positive: results free of train/test leakage and duplicate weighting; one campaign for all three papers with the 20
  paper-1 strategies; two accounts can run in parallel with complete tables per part; per-round predictions allow
  any later re-scoring; no superseded-leaf machinery to maintain; `results/` is purely append-only again (no sanctioned
  move).
- Negative / cost: about 96 estimated GPU hours (the VAAL/WAAL figures are FLOP estimates, the others come from the
  dataset with duplicates, whose pool and test set are larger) and about 43 GB of checkpoints (459 runs x ~95 MB) on
  the two Drives, plus zips of the same size if the zip cell is used. The crossing comparisons listed in Decision 5
  mix two runtime sessions (KI-43): report seed means and treat gaps within the session-to-session variation as ties.
- The thesis analysis scripts read `files_config/campaign/manifest.json`; that file now describes the new campaign
  (same group ids for every pre-existing group, new ids for the DeepAL+ groups, new roots). Tables of the first
  campaign are reproduced from git history (commit d668673) and the local reports under `docs/results/campaign/`.
- Every executed number in `.specs/experiments/campaign.md` is history of the first campaign on the dataset with
  duplicates and is superseded for the papers by this campaign.
- Follow-up: execute (`COLAB_RUNBOOK.md` section 6), merge, `make campaign-verify-all` (expect `OK=459`, seed audit PASS),
  `make campaign-report-all`; then re-derive every thesis number from the new reports.
