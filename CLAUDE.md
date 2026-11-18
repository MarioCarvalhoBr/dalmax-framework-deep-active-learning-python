# CLAUDE.md

Operational guide for Claude Code sessions in this repository.

## What this project is

DalMax is a PhD research lab (UFMS) for **Deep Active Learning applied to UAV
weed recognition**. The main contribution is **RNHAL**: a randomized-network
spatio-spectral representation (SSRAE) plus hierarchical k-means batch selection.
Entry point: `tools/trainer.py` (a thin shim calling `dalmax.cli.main()`; moved from the repo root into `tools/` on 2026-09-29, ADR 0010; renamed from the historical `demo.py` on 2026-08-23; since Phase 2 —
see below). All Python source lives in one package, `dalmax/` — `core/` and
`utils/` (the old two-package split) no longer exist, having been fully
consolidated in Phase 4 (see `.specs/architecture/current-state.md` and ADR
0002's final amendment).

## Source of truth

Specs, not this file, are authoritative for architecture, protocol, and status:

- [`.specs/00-overview.md`](.specs/00-overview.md) — what DalMax is, method summary, current status.
- [`.specs/README.md`](.specs/README.md) — index of all specs and the spec-sync contract.

If code, this file, and `.specs/` disagree, treat `.specs/` as correct and fix the
drift (see spec-sync rule below) rather than trusting stale prose here.

## Setup & commands

For the lab machine specifically (one-time setup, dataset transfer, campaign
launch/monitoring, results collection), see
[`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) — the commands below are the quick
reference; that file is the step-by-step operator guide. For Google Colab Pro
(single GPU, session-limited; hybrid local-disk + Drive-symlink layout), see
[`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md) — or run
[`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb) directly,
the runnable notebook generated to mirror it cell-for-cell.

```bash
poetry install       # or: make setup — Poetry-only; the pip/requirements.txt fallback was retired
make lint             # ruff check
make format           # ruff format
make test             # pytest, fast tests only
make smoke            # true end-to-end micro-dataset run (tools/trainer.py, CPU) + fast tests

# Example run (unchanged CLI, now routed through dalmax.cli):
poetry run python tools/trainer.py --dir_results results/dalmax1/ --params_json files_config/campaign/params_paper1.json \
    --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda

# Example run using the new generic RepresentationStrategy + embedding/selection config
# (see .specs/experiments/ablation-study.md for the exact params JSON syntax):
poetry run python tools/trainer.py --dir_results results/scratch/ --params_json files_config/ablations/rnhal/rep_spatial.json \
    --dataset_name DANINHAS --strategy_name RepresentationStrategy \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda

# Inference tools (new 2026-08-23, ADR 0006), consuming a trainer.py-written
# saved_model.pth (dalmax-checkpoint format, dalmax/models/checkpoint.py):
poetry run python tools/loader.py --model results/dalmax1/.../saved_model.pth       # inspect a checkpoint
poetry run python tools/predict.py --model results/dalmax1/.../saved_model.pth \
    --dir DATA/daninhas_full_fixed/test/DATASET_GRAMINEA --out results/predictions/ # batch prediction
poetry run python tools/predict.py --model results/dalmax1/.../saved_model.pth \
    --image path/to/one_image.jpg                                            # single-image prediction
poetry run python tools/gui.py                                                     # tkinter mini-app

# Environment checks (see LAB_RUNBOOK.md / COLAB_RUNBOOK.md; hardware-agnostic, ADR 0010):
make lab-setup                        # poetry install + torch/CUDA visibility check
make lab-check GPU=0                  # one short real-data GPU run (n_round=1) before a full batch

# The campaign -- the ONLY path for the final execution of all three papers. Since 2026-11-11 (ADR 0016) the
# manifests describe the CLEAN RE-RUN of everything on DATA/daninhas_full_fixed (KI-46), prepared, not executed:
# main manifest 297 runs / 99 groups (incl. the 3-run WAAL control) -> results/campaign_fixed/, KMH sweep 165 / 55
# -> results/campaign_kmh_fixed/; those 462 runs were executed and verified (2026-11-15). Since ADR 0017 (2026-11-15)
# the main manifest also has the part texhal_q13 (24 runs, part_1_gpu, prepared, not executed): 321 / 107, 486 in total;
# since ADR 0018 (2026-11-15) the KMH sweep also has the part kmh_wide (39 runs, part_2_gpu, prepared, not executed):
# 204 / 68, part_2_gpu 216 / 72, 525 in total, tag campaign-fixed-v4; since ADR 0019 (2026-11-15) also the part kmh_norm
# (27 runs, part_1_gpu, prepared, not executed; texhal_q13 executed with v3): 231 / 77, part_1_gpu 336 / 112, 552 in
# total, tag campaign-fixed-v5 (account A only; account B finishes kmh_wide with v4); since ADR 0020 (2026-11-16) the
# main manifest also has the parts descriptors_nq (12 runs) and seeds_ext (24 runs, seeds 4-6), both part_2_gpu,
# prepared, not executed: 357 / 119, part_2_gpu 252 / 84, 588 in total, tag campaign-fixed-v6 (account B only;
# config-only relative to v5);
# since ADR 0021 (2026-11-16) the main manifest also has the part kmh_confirm (27 runs, seeds 4-6, part_1_gpu, prepared,
# not executed; kmh_norm executed and verified, 552 OK): 384 / 128, part_1_gpu 363 / 121, 615 in total, tag
# campaign-fixed-v7 (account A only; config-only relative to v6; account B stays on v6);
# two GPU parts, one per Colab account (see .specs/experiments/campaign.md and COLAB_RUNBOOK.md section 6):
make campaign-list-gpu-part GPU_PART=part_1_gpu     # both manifests in plan order + est. GPU hours (account B: part_2_gpu)
make campaign-space-gpu-part GPU_PART=part_1_gpu    # free space vs remaining runs x 96 MB x 1.2 (exit 4 when short; read only)
make campaign-pilot GPU_PART=part_1_gpu             # first: seed 1 of the part's VAAL/WAAL (+ RandomSampling) at its largest budget
make campaign-pilot-report GPU_PART=part_1_gpu      # read only: wall time vs estimate, strategy_diagnostics, degeneracy flags
make campaign-run-gpu-part GPU_PART=part_1_gpu      # cheap groups, ablations, KMH sweep, then BIM/VAAL/WAAL(+control)/DeepFool; resume = re-run
                                                    #   (refuses to start / stops with exit 4 when the Drive is short of space)
make campaign-verify-gpu-part GPU_PART=part_1_gpu   # OK/INCOMPLETE/MISSING (integrity problems = INCOMPLETE) + git_commit audit
                                                    #   (WARN only) + seed audit (image paths) + data-order audit (0/1/2)
make campaign-verify-all                            # after merging both parts locally: expect OK=552 (OK=486 + MISSING=66 = 39 kmh_wide + 27 kmh_norm
                                                    #   until they run), up to four commits (v2, v3, v4, v5: expected WARN, ADR 0017/0018/0019), both audits
                                                    #   PASS, + the 3 intentional KMH duplicate pairs (SAME/DIFFERENT, informational);
                                                    #   since ADR 0020: OK=588 (local tree 2026-11-16: OK=525, MISSING=63 = 27 kmh_norm + 12
                                                    #   descriptors_nq + 24 seeds_ext), up to five commits (v2..v6), seed audit over seeds 1-6;
                                                    #   since ADR 0021: OK=615 (local tree 2026-11-16: OK=552, MISSING=63 = 36 descriptors_nq
                                                    #   + seeds_ext + 27 kmh_confirm), up to six commits (v2..v7); seeds 4-6 compared across
                                                    #   both accounts (kmh_confirm on A, seeds_ext on B)
make campaign-report-all                            # docs/results/campaign_fixed/ + docs/results/campaign_kmh_fixed/
make campaign-list                                  # main manifest alone; PART=paper1|paper1_control|upper_bound|rnhal|texhal|texhal_q17|hier_transfer|texhal_q13
                                                    #   |descriptors_nq|seeds_ext (comma-separated ok) or PART=part_1_gpu|part_2_gpu
                                                    #   (since ADR 0021 also PART=kmh_confirm)
make campaign-list MANIFEST=files_config/campaign/manifest_kmh_sweep.json     # the KMH sweep alone (ADR 0012); PART=kmh_nq100|kmh_nq50|kmh_nq10|kmh_wide|kmh_norm
make campaign-list MANIFEST=files_config/campaign/manifest_control_ki43.json  # KI-43 control (prepared, not run)
make campaign-smoke-gpu-part GPU_PART=part_2_gpu    # micro plan of one GPU part on CPU (seed 1), skips AdversarialBIM/DeepFool
make campaign-smoke / campaign-smoke-kmh            # micro main manifest / micro KMH sweep on CPU (seed 1)
make dataset-check                                  # build_daninhas_full_fixed.py --check (files == MANIFEST.csv, layout, no repeated MD5)
make gpu-time-estimates                             # rebuild files_config/campaign/gpu_time_estimates.json from the first campaign's trees

# Colab targets (see COLAB_RUNBOOK.md for the full notebook-cell guide):
make colab-setup                      # DRIVE_ROOT required: build/reuse daninhas_full_fixed.zip, count + MD5 check, symlink results/ -> Drive
make colab-check                      # one short real-data GPU run into results/colab_check/ (GPU 0)
```

## Working mode: multi-agent with model delegation (standing policy)

The main session (Claude Fable 5.2) orchestrates: it plans, briefs, reviews and decides, and
delegates execution to subagents by default. The policy is quality-first (revised 2026-11-10):
implementation, review, experiment audit and thesis prose → `opus` (Claude Opus 5.5; agents
`implementer`, `code-reviewer`, `experiment-auditor`, `paper-liaison`); documentation sync and
wide read-only sweeps → `sonnet` (agent `spec-keeper`); purely mechanical work (renames, running
lint/tests) → `haiku` (agent `mechanic`). When in doubt, go one tier up.
This is a **standing instruction** — it counts as the user having asked for
subagent use every time, no need to re-confirm. Full policy:
[`.claude/rules/model-delegation.md`](.claude/rules/model-delegation.md).
Agent roles: [`.claude/agents/`](.claude/agents/) (`code-reviewer`, `implementer`,
`mechanic`, `experiment-auditor`, `spec-keeper`, `paper-liaison`, `visual-analyst`, `math-verifier`).

**Spec-sync obligation**: any change to behavior, CLI, params schema, strategy
registry, or experiment protocol MUST update the corresponding `.specs/` file in
the same task, and new architectural decisions get an ADR in `.specs/adr/`. See
[`.claude/rules/spec-sync.md`](.claude/rules/spec-sync.md).

## Non-negotiables

- Code quality target (registries over if/elif, no hardcoded dataset/seed/paths,
  explicit cache keys, English-only, type hints on new code):
  [`.claude/rules/code-quality.md`](.claude/rules/code-quality.md)
- Reproducibility (params JSON + CLI + seed + git commit fully determine a run;
  never reuse an embedding cache across dataset/Q/variant):
  [`.claude/rules/reproducibility.md`](.claude/rules/reproducibility.md)
- Data safety (`DATA/` immutable, `results/` append-only, no secrets/large
  binaries committed): [`.claude/rules/data-safety.md`](.claude/rules/data-safety.md)
- Git workflow (conventional commits, small focused commits, never push without
  confirmation): [`.claude/rules/git-workflow.md`](.claude/rules/git-workflow.md)

## Current phase & next milestone

Refactor plan: [`.specs/architecture/refactor-plan.md`](.specs/architecture/refactor-plan.md).

- **Phase 1 — Safety net** (done): tests, golden-run capture, CI green.
- **Phase 2 — Core refactor** (done, 2026-08-23): config layer
  (`dalmax/config/`), embedding provider abstraction (SSRAE/VCTex/ResNet-ImageNet)
  with keyed cache (`dalmax/embeddings/`), selection module abstraction
  (`dalmax/selection/`), strategy/dataset/model registries (`dalmax/{query_strategies,
  data,models}/registry.py`), seed-propagation audit (`dalmax/seeding.py`), macro-F1
  metrics, `run_metadata.json`. `trainer.py` (historical `demo.py`) now routes through `dalmax.cli.main()`.
- **Phase 3 — Ablations** (executed 2026-08-26, on Google Colab Pro (one GPU; the GPU model is in each run's `run_metadata.json`), not the lab
  machine): all 33 runs (11 configs × 3 seeds) completed, zero failures, ~5h15 wall-clock. Weighted
  and macro F1 recorded in `.specs/experiments/ablation-study.md`'s §6.1/§6.2/§6.3 run tables and
  "Execution record" section; aggregated tables committed at `docs/results/ablation_tables/`; paper
  text drafted by `paper-liaison` (`paper_drafts/ablation_section.tex`) and applied to the paper
  under revision.
- **Phase 4 — Polish** (done, 2026-08-23, branch `refactor/phase-4-package`):
  `core/`/`utils/` physically moved into `dalmax/` (models, query strategies, data
  loaders, vendored tools, reporting scripts), and the code Phase 2 had made dead
  but not removed was deleted (`utils/orchestrator.py`, the four superseded
  strategy files, `utils/data.py`, dead scratch files) — see
  [`.specs/architecture/current-state.md`](.specs/architecture/current-state.md)
  and `refactor-plan.md` Phase 4 for the itemized move/delete list. Still
  outstanding: a real lab-machine smoke run post-move.
- **Checkpoint fix + inference tools** (done, 2026-08-23, branch `feat/model-io-tools`, ADR 0006):
  fixed the confirmed `DeepLearning.save_model`/`load_model` bug (it saved the model *class*, not
  the trained weights — every `saved_model.pth` from before this fix, including
  `results/dalmax1/`/`results/dalmax2/`, is unrecoverable, see
  [`.specs/quality/known-issues.md`](.specs/quality/known-issues.md) KI-22) via a new
  `dalmax-checkpoint` format (`dalmax/models/checkpoint.py`); added `dalmax/inference/`
  (`predict.py`/`loader.py`/`gui.py`, since 2026-09-29 in `tools/`); renamed the historical `demo.py` → `trainer.py`
  (pure rename, `git mv`, no behavior change).
- **Cleanup + hardware-agnostic** (2026-09-29, branch `chore/cleanup-tools`, ADR 0010): CLIs moved to
  `tools/`, legacy ablation/benchmark scripts and Makefile targets removed (the campaign is the only
  run path), campaign renamed to a hardware-neutral `campaign`, no GPU model named anywhere in code/docs
  (only captured at runtime in `run_metadata.json`), review follow-ups (best-effort determinism
  record, complete-leaf resume, seed-audit WARN, `--exclude-strategy`).

- **Campaign executed** (2026-09-30 to 2026-10-05 on Google Colab Pro, verified 2026-11-01; part
  `texhal_q17` executed 2026-11-01, verified 2026-11-02; part `hier_transfer` executed 2026-11-11 UTC, verified
  2026-11-10 local, ADR 0013): 222/222 runs OK, seed audit PASS for seeds 1-3;
  execution record and every result table in
  [`.specs/experiments/campaign.md`](.specs/experiments/campaign.md); reports in
  `docs/results/campaign/` (local-only since `docs/*` is gitignored; a copy is kept under
  `thesis_drafts/final_thesis/context/campaign-results/`; since 2026-11-15 the result trees `results/campaign/` and
  `results/campaign_kmh/` live under `local_assets/results_old/`, without checkpoints, see the bullet at the end of this
  list). TexHAL's canonical config is `Q=17` (ADR 0011
  accepted by the author 2026-11-01, advisor review pending; KI-40 resolved): its 6.2/6.3 ablations were
  re-run in `texhal_q17` (best 6.2 row `L=3 [300,100,50]` 0.8842; TexHAL Q=17 0.8745 edges the best
  paper-1 method, below RNHAL 0.8805). Whether RNHAL's canonical hierarchy becomes
  `L=4 [800,600,200,100]` is open (pending advisor). **Note 2026-11-15: ADR 0017 changed TexHAL's canonical config to
  VCTex `Q=13`** on the clean campaign's evidence (see the ADR 0017 bullet at the end of this list).
- **Thesis writing** (started 2026-11-01): `thesis_drafts/final_thesis/` (gitignored, own git repo,
  ReportBuilder harness), plan in `thesis_drafts/WRITING_PLAN.md`. Found 2026-11-02; author decision
  2026-11-02: runs kept as executed, behaviour disclosed in paper 1, no re-run (fix deferred to a future campaign): KI-41 (no dropout layer, so the MC-dropout
  strategies and BALD are degenerate) and KI-42 (`KMeansSampling`/`KCenterGreedy` use an untrained
  embedding layer), see `.specs/quality/known-issues.md`. **Reversed 2026-11-11 (author decision, ADR 0014)**: KI-41,
  KI-42 and KI-45 (AdversarialBIM ties broken in pool order) are fixed in code and the 63 affected runs are prepared as
  the parts `paper1_rerun_a`/`paper1_rerun_b` (two Colab accounts in parallel, `COLAB_RUNBOOK.md` section 6d; not
  executed yet).
- **KMH parameter sweep** (prepared 2026-11-07, ADR 0012; stage 1 executed 2026-11-07 to 2026-11-08 and verified locally 135/135, after the 6 `init_random` runs were re-run following KI-44: empty clusters under random k-means init crashed the hierarchical sampling, fixed in the vendored `hierarchical_sampling.py`): 45 groups / 135 runs at nq
  10/50/100 (hierarchy shapes incl. budget-matched, k-means init, leaf sampling, resampling) plus stage 2 (prepared
  2026-11-08, executed 2026-11-09 to 2026-11-10 and verified locally 165/165, nq100 only, 10 groups / 30 runs: deeper hierarchies, coarse level below budget, resampling on L=4;
  total 165 runs / 55 groups; winner `hier_L2_matched__no_resampling` 0.8819, so paper 1 reports KMH per budget: nq10 `hier_L3_matched__no_resampling` 0.6705, nq50 `hier_L3_matched` 0.8264, nq100 0.8819), manifest
  `files_config/campaign/manifest_kmh_sweep.json`, results `results/campaign_kmh/`; paper 1's KMH reference changed
  accordingly (thesis pendencia P17, author decision 2026-11-08). Runbook: `COLAB_RUNBOOK.md` section 6b.

- **DeepAL+ strategies** (2026-11-11, ADR 0015): eight new paper-1 strategies (`VarRatio`, `MeanSTD`, `KCenterGreedyPCA`,
  `BadgeSampling`, `CEALSampling`, `LossPredictionLoss`, `VAAL`, `WAAL`), optional `strategy_args` params block,
  `isolated_torch_seed` for auxiliary networks.
- **Clean re-run on `daninhas_full_fixed`** (prepared 2026-11-11, branch `feat/deepalplus-fixed-dataset`, ADR 0016; not
  executed): KI-46 (byte-identical duplicates in `DATA/daninhas_full`, including 171 images in both train and test) is
  resolved by the new canonical dataset `DATA/daninhas_full_fixed` (7,230 train / 1,008 test); EVERYTHING is re-run
  into new roots (`results/campaign_fixed/`, `results/campaign_kmh_fixed/`), paper 1 now has 20 strategies + KMH, the
  ADR 0014 re-run parts / archive script / superseded-leaf guard were retired unexecuted, every run also saves
  `round_predictions.json` / `round_queries.json`, and the work is split into two GPU parts for two Colab accounts
  (`part_1_gpu` 285 runs, about 49.9 estimated GPU h since the WAAL control; `part_2_gpu` 177 runs, about 48.8 h). Every executed number above
  (first campaign, KMH sweep, hier_transfer) was obtained on the dataset with duplicates and is superseded for the papers.
- **Pre-flight fix KI-47** (2026-11-12, ADR 0016 amendment, before any run of the clean campaign): the loaders listed
  each class folder with an unsorted `os.listdir`, whose order differed between Colab sessions, so in the first
  campaign the same seed started from other images in every later session, and the Drive-resident embedding cache
  (keyed on pool indices) served the KMH sweep, `texhal_q17`, `hier_transfer` and paper-1 KMH nq50/nq10 the features
  of other images (the evidence behind ADR 0012/0013, and the `texhal_q17` results after ADR 0011, must be re-read on
  the clean campaign; very likely the true cause of KI-43). Fixed: sorted file names (`list_image_names`,
  `dalmax/data_order.py`), `run_metadata.json` `data_order` hashes, an order-keyed and path-validated cache with
  atomic writes, the seed audit by round-0 image paths plus a data-order audit in `campaign verify`; also a Drive
  quota gate (`campaign run` exit 4), a pilot (`make campaign-pilot`) and CEAL estimated at 3x a plain run.
- **Pre-flight fix KI-48** (2026-11-12, ADR 0016 amendment, second pass): a leaf with a truncated/empty JSON, a wrong
  round count or an empty checkpoint is INCOMPLETE (re-run on resume); leaves are looked up only under the dataset
  folder of the group's params (a plan leaf under another folder FAILs the data-order audit); `verify`/`pilot-report`
  print the distinct `git_commit` values (WARN on more than one); the notebook checks out `GIT_REF` (tag
  `campaign-fixed-v2` since 2026-11-12; v1 predates the two items below) detached on every session instead of pulling
  `main`, and surfaces a non-zero run exit.
- **WAAL control + dataset metadata layout** (2026-11-12, branch `feat/waal-control-dataset-layout`, ADR 0015 and ADR
  0016 amendments, before any run): `WAALRandomQuery` = WAAL's training unchanged (same `strategy_args.WAAL` block,
  trainer generator and round-0 model for a seed) with a uniform random query (`derive_seed(seed,
  "waal_random_query")`), so a WAAL gain over `RandomSampling` can be split into training (about 280 SGD steps per
  round against 10 to 40) and selection; part `paper1_control` at nq100 (3 runs, `part_1_gpu`, stage 5), its own table
  `paper1_waal_control_nq100`, never a benchmark row (paper 1 stays 20 strategies + KMH). `DATA/daninhas_full_fixed`
  gets `logs/build_history.txt` + `logs/source_manifest.csv` (provenance), a top-level `README.txt`/`MANIFEST.csv`
  describing the dataset as it is (`split,class,file,md5`), 8,242 files; `--check` now also compares the files on disk
  with `MANIFEST.csv`; the one-off `--migrate-metadata` converts the existing folder (images untouched).
- **First-campaign archival (2026-11-15, author, by hand; not a precedent)**: `results/campaign/`,
  `results/campaign_kmh/`, `results/cache/`, `results/colab_check/` and `results/campaign_tables/` were moved to
  `local_assets/results_old/` (gitignored) after deleting the 387 `saved_model.pth` (Drive backup `results_old` kept);
  metrics, predictions, logs and `run_metadata.json` remain, so `campaign verify` there would say INCOMPLETE (read-only
  record). `results/` holds only `campaign_fixed/`, `campaign_kmh_fixed/`, `logs/`, `original_indices.txt`, `smoke/`.
  The never-touch-`results/` rule for agents and code is unchanged (`.claude/rules/data-safety.md`).
- **TexHAL canonical `Q=13` (2026-11-15, branch `feat/texhal-q13`, ADR 0017; prepared, not executed)**: the clean
  campaign (462 runs, tag `campaign-fixed-v2` = `464a5cc`) was executed, merged locally and verified (OK=462, one
  commit, seed and data-order audits PASS, the 3 KMH duplicate pairs SAME). Its paper-2 6.1 (nq100 weighted F1):
  VCTex `Q=13` 0.8341, `Q=[5,17]` 0.8038, `Q=5` 0.7766, `Q=17` 0.7703; KMH 0.7921. Author decision: TexHAL's canonical
  configuration is `Q=13` (`texhal/6_1/rep_q13`; supersedes ADR 0011; Q chosen on the test set, disclosed). New part
  `texhal_q13` = the `texhal_q17` 6.2/6.3 groups + its two `hier_transfer` shapes with `q=[13]` (8 groups / 24 runs,
  `files_config/ablations/texhal_q13/`, `part_1_gpu`, stage 2); aliases `texhal_q13/6_2/hier_ref`, `6_3/stage_full` ->
  `texhal/6_1/rep_q13`. Main manifest 321 / 107, `part_1_gpu` 309 / 103, 486 in total; the 462 executed groups keep
  their identity (digest test). Paper 2's `6_2`/`6_3` come from `texhal_q13`, the executed Q=17 tables are
  `6_2_q17`/`6_3_q17`; 6.1 rows relabelled `Q=13 (TexHAL)` / `Q=17`; the TexHAL comparison row of papers 2 and 3 is
  "TexHAL (VCTex Q=13 + hierarchical)". Notebook: `GIT_REF` `campaign-fixed-v3`, `EXPECTED` part_1_gpu (309, 103),
  `ZIP_ONLY_PART`; the code-version audit of account A then lists two commits (expected WARN). `COLAB_RUNBOOK.md` 6e.
- **KMH wide-top exploration (2026-11-15, branch `feat/kmh-wide`, ADR 0018; prepared, not executed)**: on the clean
  campaign paper 1's KMH is below WAAL at every query size (best KMH sweep configuration vs WAAL: nq100 0.7921 vs
  0.8077, nq50 0.7146 vs 0.7874, nq10 0.5627 vs 0.5793). New part `kmh_wide` of the KMH sweep manifest: 13 groups / 39
  runs, ids `kmh_wide/nq{N}/<config>`, results `results/campaign_kmh_fixed/wide/nq{N}/<config>/`, `part_2_gpu` (account
  B: its Drive holds the ImageNet feature cache, whose key does not depend on `n_query`), stage 3; (a) wide top with
  the last level = n_query at nq10/nq50 (`[600,200,nq]`, `[600,nq]`), (b) wider top at nq 10/50/100 (L=1 `k=1200`,
  `k=2400`; L=3 `[1200,400,nq]`); sample sizes by the sweep's 0.05 rule, other knobs default; params
  `files_config/ablations/kmh_sweep/nq{N}/`. Estimates by a linear fit in the total number of clusters (9.2 GPU h).
  Report tables `kmh_wide_nq{100,50,10}`. Sweep 204 / 68, `part_2_gpu` 216 / 72, 525 in total; the v3 groups keep their
  identity (digest test). Engine: only `kmh_wide` in `PARTS`/`PHASE_LABELS`. Notebook: `GIT_REF` `campaign-fixed-v4`,
  `EXPECTED` part_2_gpu (216, 72), `ZIP_ONLY_PART = "kmh_wide"`; account B's audit lists two commits (v2, v4), the
  merged one up to three. Paper 1's KMH row (thesis rule) then ranges over `kmh_nq{N}` + `kmh_wide`; a configuration
  chosen from it is selected on the test set and is confirmed on seeds 4-6 with RandomSampling and WAAL before any claim
  (planned, not built). `COLAB_RUNBOOK.md` 6f.
- **KMH feature post-processing (2026-11-15, branch `feat/kmh-feature-norm`, ADR 0019; prepared, not executed)**: KMH
  feeds raw 2048-d ImageNet vectors to a Euclidean k-means; under the reference hierarchy at nq100 another descriptor
  moved the score by 4 to 7 points (VCTex Q=13 0.8341, SSRAE 0.8608, ImageNet 0.7921). New optional params key
  `embedding.postprocess` (`dalmax/embeddings/postprocess.py`, registry `POSTPROCESS_REGISTRY`): `none` (default,
  identity), `l2`, `zscore`, `pca_l2` + `pca_dim`, fitted once per run on the initial unlabeled pool, applied by
  `RepresentationStrategy` after the cache load and the variant slice (cache unchanged), no RNG draw; the default path is
  tested identical to tag `campaign-fixed-v4` (spy vs the v4 class, golden runs, end-to-end micro run with and without
  an explicit `"none"`, pinned v4 digest of the 175 groups). New part `kmh_norm` of the KMH sweep: 9 groups / 27 runs,
  the reference hierarchy with `l2` / `zscore` / `pca_l2` (64) at nq 10/50/100, ids `kmh_norm/nq{N}/{l2,zscore,pca64_l2}`,
  results `results/campaign_kmh_fixed/norm/nq{N}/<variant>/`, `part_1_gpu` (account A), stage 3; estimates = the measured
  reference (4.2 GPU h); tables `kmh_norm_nq{100,50,10}` with the executed reference as an alias row. Sweep 231 / 77,
  `part_1_gpu` 336 / 112, 552 in total. KI-50 kept on purpose: round 0 must equal the executed KMH runs (0.3839 / 0.4783
  / 0.2469). Notebook: `GIT_REF` `campaign-fixed-v5` (unlike v3/v4 it changes trainer code, behind the new key),
  `EXPECTED` part_1_gpu (336, 112), `ZIP_ONLY_PART = "kmh_norm"`; account A's audit lists three commits (v2, v3, v5), the
  merged one up to four. Same seeds 4-6 confirmation rule. `COLAB_RUNBOOK.md` 6g.
- **Descriptors at small query sizes + seeds 4-6 (2026-11-16, branch `feat/descriptors-small-nq`, ADR 0020; prepared, not
  executed)**: two config-only parts of the main manifest, both `part_2_gpu` (account B, idle). `descriptors_nq` (pendencia
  P36): RNHAL (`rnhal/6_3/stage_full` params, 0.8608 at nq100) and TexHAL (`texhal/6_1/rep_q13` params, 0.8341) at
  n_query 50 and 10, 4 groups / 12 runs, ids `descriptors_nq/{rnhal,texhal_q13}/nq{50,10}`, results
  `results/campaign_fixed/descriptors_nq/`, stage 2, 1.9 GPU h (nq100 run x KMH ratio), tables `descriptors_nq{50,10}`
  (paper 1's report, with the executed paper-1 WAAL, BALD, BADGE, KMH, Random of that size). `seeds_ext` (three-seed
  criticism): the 8 headline methods at nq100 (Random, BALD, BADGE, KMH, upper bound, RNHAL, TexHAL, WAAL), exact copies
  of their seeds-1-3 groups on seeds 4, 5, 6, 8 groups / 24 runs, ids `seeds_ext/<method>`, results
  `results/campaign_fixed/seeds_ext/`, 4.1 GPU h (counterpart's entry), table `seeds_ext_nq100` (paper 3's report; never
  merged into a seeds-1-3 row). Engine (`dalmax/campaign.py` only, nothing a trainer run executes): part names;
  redundancy = same computation on a common seed; pilot only among groups that run seed 1 (`part_2_gpu`'s pilot stays the
  executed nq50 one). The seed audit needs no change (each seed's runs vs the first of that seed). Main 357 / 119,
  `part_2_gpu` 252 / 84, 588 in total; the 184 v5 groups keep their identity (pinned v5 digests), sweep manifests
  byte-identical. Notebook: `GIT_REF` `campaign-fixed-v6` (account B only; A stays on v5 until `kmh_norm` is merged),
  `EXPECTED` part_2_gpu (252, 84), `ZIP_ONLY_PART = "descriptors_nq+seeds_ext"` (several parts joined by `+`, one zip
  `part_2_gpu__descriptors_nq__seeds_ext.zip`, 36 checkpoints); account B's audit lists v2, v4, v6. `COLAB_RUNBOOK.md` 6h.
- **KMH l2 confirmed on seeds 4-6 (2026-11-16, branch `feat/kmh-confirm`, ADR 0021; prepared, not executed)**: `kmh_norm`
  executed and verified (552 OK; seeds 1-3, test set: nq100 `l2` 0.8128, `pca64_l2` 0.8155, reference 0.7921, WAAL
  0.8077; nq50 `l2` 0.7086, reference 0.5860, WAAL 0.7874, `pca64_l2` 0.5589 with one seed collapsed; nq10 `l2` 0.4493,
  reference 0.4909, WAAL 0.5793). The variant to confirm was fixed before any seed 4-6 result: `l2` only, at nq 100/50/10
  (nq10 included although it lost; `pca64_l2` unstable, not confirmed). Config-only part `kmh_confirm` of the MAIN
  manifest (table rows cannot cross manifests and the nq100 rows are `seeds_ext` runs; Random/WAAL are main-manifest
  groups; one zip folder), `part_1_gpu` (account A, idle): 9 groups / 27 runs on seeds 4, 5, 6, exact copies of
  `kmh_norm/nq{100,50,10}/l2` (stage 3), `p1/KMH/nq{50,10}` and `p1/RandomSampling/nq{50,10}` (stage 1), `p1/WAAL/nq{50,10}`
  (stage 5); ids `kmh_confirm/nq{N}/{kmh_l2,kmh_reference,RandomSampling,WAAL}`, results
  `results/campaign_fixed/kmh_confirm/`; the nq100 reference/Random/WAAL are aliases of `seeds_ext/{KMH,RandomSampling,WAAL}`
  (no second run). Tables `kmh_confirm_nq{100,50,10}` in paper 3's report next to `seeds_ext_nq100` (single-consumer rule
  kept). Estimates from the counterparts' clean-campaign wall times: 8.5 GPU h (WAAL 6.0 h). Account A computes the
  ImageNet cache of seeds 4-6 once per seed (first KMH run, key independent of n_query). Engine: part list only; pilot
  unchanged (all SKIP); the seed audit pools a seed's runs across parts and accounts. Main 384 / 128, `part_1_gpu`
  363 / 121, `part_2_gpu` 252 / 84 unchanged, 615 in total; the 196 v6 groups keep their identity (pinned v6 digests).
  Notebook: `GIT_REF` `campaign-fixed-v7` (account A only; B stays on v6 until its 36 runs are merged), `EXPECTED`
  part_1_gpu (363, 121), `ZIP_ONLY_PART = "kmh_confirm"` (27 checkpoints); account A's audit lists v2, v3, v5, v7.
  `COLAB_RUNBOOK.md` 6i.
- **Campaign closed (2026-11-17)**: the author closed the campaign at 615 runs, no further runs. `make
  campaign-verify-all` on 2026-11-17: `OK=615 INCOMPLETE=0 MISSING=0`, six expected commits (v2 `464a5cc` 462 leaves, v4
  39, v6 36, v5 27, v7 27, v3 24), seed audit PASS for seeds 1-6, data-order audit PASS; `make campaign-report-all`
  regenerated (main 22 tables, 128/128 groups; KMH sweep 13 tables, 77/77). The parts the bullets above call "prepared,
  not executed" (`texhal_q13` v3, `kmh_wide` v4, `kmh_norm` v5, `descriptors_nq` + `seeds_ext` v6, `kmh_confirm` v7) were
  all executed on the two Colab accounts and verified (2026-11-16/17); outcomes in ADR 0018-0021 ("Outcome" sections)
  and `.specs/experiments/campaign.md`. Headlines: no wide top beats the earlier KMH best or WAAL; `l2` features lift KMH
  but are confirmed on seeds 4-6 at nq50 only (NOT at nq100, nq10); six-seed nq100 comparison RNHAL 0.8660, TexHAL
  0.8311, WAAL 0.7679, KMH 0.7379, full supervision 0.8611. Author decisions 2026-11-17/18: headline comparison at nq100
  on six seeds, ablations and the 20-strategy benchmark on three; paper 1's KMH stays the reference-hierarchy baseline
  with `l2` as a documented variant; descriptors at nq 50/10 are a robustness analysis. Thesis release v3.0.0
  (2026-11-18) is written from these 615 runs.

Next milestone (revised 2026-11-18): no run is outstanding (the commands and "prepared, not executed" notes above are the
preparation-time record). What remains is the advisor's reading of the thesis and papers, and the separately outstanding
lab-machine smoke run confirming the Phase 4 and `tools/` moves. Historical text of the previous milestone follows,
kept for the record: execute the clean re-run (`COLAB_RUNBOOK.md` section 6), then rewrite the thesis numbers from its reports; advisor review of the ablation section
(`paper_drafts/ablation_section.tex`, weighted-primary/macro-secondary framing still pending
confirmation, see `.specs/experiments/ablation-study.md`), of the TexHAL `Q=17` decision (already executed and tabulated; superseded by `Q=13`, ADR 0017, whose 24-run `texhal_q13` part was executed with v3, `COLAB_RUNBOOK.md` section 6e; next to execute: the 39-run `kmh_wide` part of ADR 0018 on account B, section 6f, and the 27-run `kmh_norm` part of ADR 0019 on account A, section 6g; then the 36 runs of `descriptors_nq` + `seeds_ext` of
ADR 0020 on account B, section 6h; then the 27 runs of `kmh_confirm` of ADR 0021 on account A, section 6i) and of the RNHAL
`L=4` hierarchy; a lab-machine smoke run confirming the Phase 4 and `tools/` moves is still separately
outstanding.

## Never do

- Touch or write into `DATA/` (immutable input).
- Edit or delete anything under `results/` (append-only experiment history).
- Run a full training/experiment locally (no GPU here) — local runs are for
  `make smoke` / tiny CPU subsets only; real training happens on the lab machine
  or Colab.
- Commit `.pkl`, `.pth`, or other large/generated artifacts.
- `git push` without explicit user confirmation.

## Key facts cheat-sheet

- SSRAE hidden-layer size `Q = 13` (config-driven since Phase 2:
  `EmbeddingConfig.q`, defaulted per-extractor in `dalmax/config/loader.py`; the
  legacy hardcoded literal was in `utils/data.py::create_feature_maps_ssrae`,
  deleted entirely in Phase 4); VCTex uses `Q ∈ {5, 17}` (i.e. `Q = (5, 17)` as a
  tuple, not two runs).
- Results dir (unchanged by Phase 2/4): `{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/`
  — now also contains `run_metadata.json` (config snapshot + git commit), `results.json` gained
  `all_precision_macro`/`all_recall_macro`/`all_f1_macro`, and since ADR 0016 every run also writes
  `round_predictions.json` (test predictions of every round) and `round_queries.json` (pool indices labeled per round,
  with labels and paths); `dataset_folder` is `daninhas_full_fixed` for the clean campaign.
- **Dataset (ADR 0016)**: canonical `DATA/daninhas_full_fixed/` (7,230 train / 1,008 test images + `README.txt` +
  `MANIFEST.csv` (`split,class,file,md5`) + `logs/build_history.txt` + `logs/source_manifest.csv` = 8,242 files since
  the metadata layout of 2026-11-12, no repeated MD5; built by `scripts/datasets/build_daninhas_full_fixed.py`, the only
  sanctioned way a dataset directory is created, `.claude/rules/data-safety.md`; its `--migrate-metadata` is the one-off,
  images-untouched conversion of the existing folder authorised on 2026-11-12). `DATA/daninhas_full/` (with duplicates, KI-46) stays
  untouched as the source and as the dataset of the first campaign; micro configs keep `DATA/daninhas_micro/`.
- **New CLI flags (Phase 2)**: `--device {auto,cuda,cpu}` (default `auto`; pass
  `--device cuda` explicitly on the lab machine, don't rely on `auto` — see
  `.specs/infrastructure/execution-environments.md`), `--embedding_variant
  {full,spatial,spectral}` (SSRAE only, overrides the params JSON's `embedding.variant`).
- **Embedding cache path (Phase 2; `order` segment since 2026-11-12, KI-47)**:
  `results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__order{order_hash}__pool{hash}.pkl`
  (`dalmax/embeddings/cache.py`; the payload also stores and checks the cached images' paths, so a cache of another
  image order is never used) — the Phase 1 `results/cache/{name}_{dataset_folder}...`
  path was deleted along with `utils/data.py` in Phase 4;
  `results/features_dict_*.pkl` (original, orphaned) files may still be on disk
  but are read by nothing.
- `trainer.py --strategy_name` choices (unchanged plus one new generic name):
  `RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`,
  `SSRAEKmeansSampling`, `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`,
  `VCTexKmeansHCSampling`, **`VarRatio`**, **`MeanSTD`**, **`KCenterGreedyPCA`**, **`BadgeSampling`**,
  **`CEALSampling`**, **`LossPredictionLoss`**, **`VAAL`**, **`WAAL`** (2026-11-11, ADR 0015: the DeepAL+ ports, in paper 1
  since ADR 0016; hyperparameters in the optional `strategy_args` params block), **`WAALRandomQuery`** (2026-11-12, ADR 0015
  amendment: the WAAL training-budget control, WAAL's training + a uniform random query, reads the `WAAL` block; a
  control, not a benchmark strategy), **`FullSupervised`** (2026-09-29 — paper-1 upper bound, *not* active
  learning: whole pool labeled, requires `--n_round 0`, `query()` raises),
  **`RepresentationStrategy`** (NEW — generic, driven
  by the params JSON's `"embedding"`/`"selection"` blocks; see
  `.specs/experiments/ablation-study.md` for exact syntax).
- **Ablation scripts, `METHOD` variable and legacy tree (retired 2026-09-29, ADR 0010)**: the
  per-GPU ablation scripts (`scripts/ablations/`, `scripts/colab/run_ablations_colab.sh`), the
  reference-benchmark scripts (`scripts/benchmark/`), `make ablations-*`/`ablation-report*`/
  `benchmark-*`/`smoke-ablations`, `dalmax/reporting/ablation_report.py` and `results_doctor` were
  deleted: the campaign manifest is the only way to run and report. The per-method configs still
  live in `files_config/ablations/{rnhal,texhal}/` (+ `micro/`) because the manifest references them
  (`files_config/ablations/README.md`); the executed 2026-08-26 RNHAL batch remains on disk at the
  legacy `results/ablations/{6_1,6_2,6_3}/` (append-only) with its committed tables under
  `docs/results/ablation_tables/`. The leaf completeness check survives as
  `dalmax/reporting/leaf_check.py::check_leaf`. Three-paper plan: `.specs/experiments/papers-roadmap.md`.
- `make lab-check`/`colab-check` use `files_config/campaign/params_paper1.json` (since 2026-09-29);
  the legacy `files_config/benchmark/params_df_gpu_{0,1}.json` are kept only for their CIFAR10 block
  (used by `tests/test_config.py`); the historical reference sweep scripts were retired
  2026-09-29 (the campaign supersedes them). They still work via the legacy `config_kmh` key (Phase 2's loader reads it as
  `selection = {method: "hierarchical", hierarchy: config_kmh}` automatically).
- **Campaign (2026-09-29, ADR 0008/0009/0010)**: `files_config/campaign/manifest.json` (generated by
  `scripts/campaign/build_manifest.py`, mirror `manifest_micro.json`) is the single source of truth for
  the one-shot re-execution of everything the three papers need: paper 1 = 12 classical strategies +
  **KMH** (hierarchical k-means over ImageNet ResNet-50 features, confirmed by the user) x nq {10,50,100}
  + the `FullSupervised` upper bound; papers 2/3 = TexHAL/RNHAL ablations at nq100 (+ 5 new RNHAL
  hierarchy rows). 222 runs / 74 groups (all 222 executed; the 18-run `texhal_q17` part ran 2026-11-01, ADR 0011; the 12-run `hier_transfer` part, ADR 0013, ran 2026-11-11 UTC and was verified 2026-11-10 local), results under `results/campaign/`; groups consumed by
  several papers live under `shared/` (`shared/kmh_nq100`, `shared/random_nq100`,
  `shared/texhal_full`) and every alias row points to the one run. `stage_no_representation` is ONE
  shared run (2026-09-01 per-paper decision superseded by ADR 0009; its only config is
  `files_config/campaign/params_kmh.json`). Engine `dalmax/campaign.py`, reports
  `dalmax/reporting/campaign_report.py`. `seed_everything` now also sets `CUBLAS_WORKSPACE_CONFIG` and
  `torch.use_deterministic_algorithms(True, warn_only=True)`; `run_metadata.json` has a `determinism` block.
  (Counts and paths in this bullet are the FIRST campaign, on the dataset with duplicates; see the next bullet.)
- **Clean re-run manifests (ADR 0016, prepared 2026-11-11, not executed)**: the committed `manifest.json` now has 297
  runs / 99 groups (paper1 189 = 20 strategies + KMH x nq {10,50,100}; paper1_control 3 = `WAALRandomQuery` nq100,
  2026-11-12; upper_bound 3 with `n_init_labeled` 7,230; rnhal 42, texhal 30, texhal_q17 18, hier_transfer 12; since ADR
  0017, 2026-11-15, also texhal_q13 24, so 321 / 107) under `results/campaign_fixed/`, and `manifest_kmh_sweep.json`
  165 / 55 under `results/campaign_kmh_fixed/` (since ADR 0018, 2026-11-15, also kmh_wide 39 under `wide/`, so 204 / 68,
  all of kmh_wide in `part_2_gpu`; since ADR 0019, 2026-11-15, also kmh_norm 27 under `norm/`, so 231 / 77, all of
  kmh_norm in `part_1_gpu`; since ADR 0020, 2026-11-16, the main manifest also has descriptors_nq 12 and seeds_ext 24
  (seeds 4-6), both `part_2_gpu`, so 357 / 119; since ADR 0021, 2026-11-16, also kmh_confirm 27 (seeds 4-6),
  `part_1_gpu`, so 384 / 128). Every group of both carries `gpu_part` (`part_1_gpu` = paper 1 nq100,
  upper bound, all ablations, sweep nq100+nq50; `part_2_gpu` = paper 1 nq50+nq10, sweep nq10) and `gpu_stage`
  (1 cheap paper-1, 2 ablations, 3 KMH sweep, 4 BIM, 5 VAAL/WAAL/WAAL control, 6 DeepFool); `--part part_N_gpu` selects by it,
  `--manifest A,B` merges manifests into one plan. Params: dropout family (4 MC-dropout + `VarRatio`, `MeanSTD`)
  `params_paper1_mc_dropout.json`, `VAAL` `params_paper1_vaal.json` (`image_size` 32), others `params_paper1.json`.
  Estimates: `files_config/campaign/gpu_time_estimates.json` (`campaign list` prints hours). Manifests list
  `required_leaf_files` (`round_predictions.json`, `round_queries.json`). `effective_signature` reads only the
  strategy's own `strategy_args` block.
- **Hierarchy transfer part `hier_transfer` (ADR 0013, prepared 2026-11-10, executed 2026-11-11 UTC, verified 222/222)**: 4 groups / 12 runs appended to
  the main manifest (`make campaign-run PART=hier_transfer`): the KMH-sweep nq100 winner shape `k=[600,100]`, `sample_sizes=[30,5]`,
  `n_resamples=0` (0.8819 with ImageNet features) and the reference shape without resampling, each with SSRAE (`rnhal/6_2/`)
  and VCTex Q=17 (`texhal_q17/6_2/`); rows of table 6.2 of papers 3 and 2. Outcome: the shape does not transfer; under `k=[600,100]` without resampling the descriptors (SSRAE 0.8592, VCTex Q=17 0.8544) are below the ImageNet features (0.8819); no canonical configuration changes. It ran on a third runtime image with a different GPU model (see `run_metadata.json`). Runbook: `COLAB_RUNBOOK.md` section 6c.
- **Fixed baselines re-run (ADR 0014, prepared 2026-11-11, never executed; operational part superseded by ADR 0016:
  the parts below, the archive script and the superseded-leaf guard were retired and the 21 groups are back in
  `paper1` of the clean campaign; kept for the record)**: the 21 groups `p1/<S>/nq{10,50,100}` of
  `AdversarialBIM`, `KMeansSampling`, `KCenterGreedy` and the four MC-dropout strategies moved from `paper1` (now 54 runs /
  18 groups) to `paper1_rerun_a` (BIM nq100/nq50 + KMeans + KCenter, 24 runs, ~4.3 GPU h) and `paper1_rerun_b` (BIM nq10 +
  MC dropout with `files_config/campaign/params_paper1_mc_dropout.json`, 39 runs, ~4.2 GPU h); same ids, results paths and
  table rows (222 runs / 74 groups unchanged). Before running, `make campaign-archive-superseded` moves the superseded
  first-execution leaves to `results/old_results_broken/` (deleted unexecuted with ADR 0016; `results/` has no
  sanctioned move any more).
- **Model `model_args` and MC dropout (ADR 0014)**: optional params-JSON block `"model_args": {"dropout_p": <[0,1)>}`
  (absent = the model's defaults, resolved `{"dropout_p": null}`); `DaninhasModelResNet50` returns the 2048-d pooled
  features as embedding (`embedding_layer` kept unused for the RNG stream/checkpoint keys); `predict_prob_dropout*` run in
  eval mode with only dropout active and raise `MCDropoutError` without dropout; `AdversarialBIM` breaks distance ties
  with `derive_seed(seed, "adversarial_ties")` (`AdversarialDeepFool` keeps its pool-order tie-break, not re-run).
- **Optional `selection.hierarchy` keys (ADR 0012)**: `n_resamples` (default 10), `init_method`
  (`kmeans++`|`random`), `resample_strategy` (`closest`|`random`), `leaf_sampling` (`random`|`closest`);
  the defaults reproduce the previous behaviour, so existing params JSONs and executed runs are unchanged.
- **Optional `embedding.postprocess` key (ADR 0019, 2026-11-15)**: `{"method": "none"|"l2"|"zscore"|"pca_l2",
  "pca_dim": <int, pca_l2 only>}` inside the `"embedding"` block; absent = `"none"` (identity, resolved
  `{"method": "none", "pca_dim": null}` in `run_metadata.json`); fitted once per run on the initial unlabeled pool, applied
  by `RepresentationStrategy` (presets and generic) after the cache load and the variant slice; never cached; fail-fast
  `ConfigError` (unknown method/key, `pca_dim` missing for `pca_l2` or set for another method, `pca_dim` above the pool
  or the feature dimension); a non-identity method adds a log line and `postprocess_*` `strategy_diagnostics`; the
  campaign signature counts it only when not `none`. Syntax: `.specs/experiments/experimental-protocol.md`.
