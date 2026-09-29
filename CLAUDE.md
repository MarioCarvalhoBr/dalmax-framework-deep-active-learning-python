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
poetry run python tools/trainer.py --dir_results results/dalmax1/ --params_json files_config/benchmark/params_df_gpu_0.json \
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
    --dir DATA/daninhas_full/test/DATASET_GRAMINEA --out results/predictions/ # batch prediction
poetry run python tools/predict.py --model results/dalmax1/.../saved_model.pth \
    --image path/to/one_image.jpg                                            # single-image prediction
poetry run python tools/gui.py                                                     # tkinter mini-app

# Environment checks (see LAB_RUNBOOK.md / COLAB_RUNBOOK.md; hardware-agnostic, ADR 0010):
make lab-setup                        # poetry install + torch/CUDA visibility check
make lab-check GPU=0                  # one short real-data GPU run (n_round=1) before a full batch

# The campaign (2026-09-29) -- the ONLY path for the final execution of all three papers
# (one manifest, no redundant runs; see .specs/experiments/campaign.md and COLAB_RUNBOOK.md):
make campaign-list                    # job table + run counts per part (192 runs / 64 groups)
make campaign-run PART=all            # PART=paper1|upper_bound|rnhal|texhal (comma-separated ok); resume = re-run
make campaign-verify                  # OK/INCOMPLETE/MISSING per job + seed audit (exit 0 ok / 1 fail / 2 warn)
make campaign-report                  # docs/results/campaign/ (tables md/tex/csv, mean confusion matrices)
make campaign-smoke                   # micro campaign on CPU (seed 1) -> results/smoke_campaign/;
                                      # skips AdversarialBIM/AdversarialDeepFool by default (58 of 64 jobs; SMOKE_EXCLUDE= to include)

# Colab targets (see COLAB_RUNBOOK.md for the full notebook-cell guide):
make colab-setup                      # verify Drive mount, build/reuse dataset zip, symlink results/ -> Drive
make colab-check                      # one short real-data GPU run into results/colab_check/ (GPU 0)
```

## Working mode: multi-agent with model delegation (standing policy)

The main session orchestrates; it delegates coding to subagents by default and
only works directly when a task is genuinely complex enough to need the top
model. Routine implementation → `sonnet` (agent: `implementer`); trivial/mechanical
work (renames, config tweaks, running lint/tests) → `haiku` (agent: `mechanic`).
This is a **standing instruction** — it counts as the user having asked for
subagent use every time, no need to re-confirm. Full policy:
[`.claude/rules/model-delegation.md`](.claude/rules/model-delegation.md).
Agent roles: [`.claude/agents/`](.claude/agents/) (`code-reviewer`, `implementer`,
`mechanic`, `experiment-auditor`, `spec-keeper`, `paper-liaison`).

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

Next milestone: advisor review of the ablation section (`paper_drafts/ablation_section.tex`,
including the weighted-primary/macro-secondary framing pending confirmation — see
`.specs/experiments/ablation-study.md`), and the full campaign execution on a GPU machine (Colab or
lab; `make campaign-run PART=all`); a lab-machine smoke run confirming the Phase 4 move and the
`tools/` move didn't break anything there is still separately outstanding.

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
  — now also contains `run_metadata.json` (config snapshot + git commit), and
  `results.json` gained `all_precision_macro`/`all_recall_macro`/`all_f1_macro`.
- **New CLI flags (Phase 2)**: `--device {auto,cuda,cpu}` (default `auto`; pass
  `--device cuda` explicitly on the lab machine, don't rely on `auto` — see
  `.specs/infrastructure/execution-environments.md`), `--embedding_variant
  {full,spatial,spectral}` (SSRAE only, overrides the params JSON's `embedding.variant`).
- **Embedding cache path (Phase 2)**: `results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{hash}.pkl`
  (`dalmax/embeddings/cache.py`) — the Phase 1 `results/cache/{name}_{dataset_folder}...`
  path was deleted along with `utils/data.py` in Phase 4;
  `results/features_dict_*.pkl` (original, orphaned) files may still be on disk
  but are read by nothing.
- `trainer.py --strategy_name` choices (unchanged plus one new generic name):
  `RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`,
  `SSRAEKmeansSampling`, `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`,
  `VCTexKmeansHCSampling`, **`FullSupervised`** (2026-09-29 — paper-1 upper bound, *not* active
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
- One params JSON per lab GPU: `files_config/benchmark/params_df_gpu_0.json`,
  `files_config/benchmark/params_df_gpu_1.json`, run
  (used by `make lab-check`/`colab-check`; the historical reference sweep scripts were retired
  2026-09-29, the campaign supersedes them) — still
  works via the legacy `config_kmh` key (Phase 2's loader reads it as
  `selection = {method: "hierarchical", hierarchy: config_kmh}` automatically).
- **Campaign (2026-09-29, ADR 0008/0009/0010)**: `files_config/campaign/manifest.json` (generated by
  `scripts/campaign/build_manifest.py`, mirror `manifest_micro.json`) is the single source of truth for
  the one-shot re-execution of everything the three papers need: paper 1 = 12 classical strategies +
  **KMH** (hierarchical k-means over ImageNet ResNet-50 features, confirmed by the user) x nq {10,50,100}
  + the `FullSupervised` upper bound; papers 2/3 = TexHAL/RNHAL ablations at nq100 (+ 5 new RNHAL
  hierarchy rows). 192 runs / 64 groups, results under `results/campaign/`; groups consumed by
  several papers live under `shared/` (`shared/kmh_nq100`, `shared/random_nq100`,
  `shared/texhal_full`) and every alias row points to the one run. `stage_no_representation` is ONE
  shared run (2026-09-01 per-paper decision superseded by ADR 0009; its only config is
  `files_config/campaign/params_kmh.json`). Engine `dalmax/campaign.py`, reports
  `dalmax/reporting/campaign_report.py`. `seed_everything` now also sets `CUBLAS_WORKSPACE_CONFIG` and
  `torch.use_deterministic_algorithms(True, warn_only=True)`; `run_metadata.json` has a `determinism` block.
