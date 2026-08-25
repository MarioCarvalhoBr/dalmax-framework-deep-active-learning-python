# CLAUDE.md

Operational guide for Claude Code sessions in this repository.

## What this project is

DalMax is a PhD research lab (UFMS) for **Deep Active Learning applied to UAV
weed recognition**. The main contribution is **RNHAL**: a randomized-network
spatio-spectral representation (SSRAE) plus hierarchical k-means batch selection.
Entry point: `trainer.py` (a thin shim calling `dalmax.cli.main()`, renamed from the historical `demo.py` on 2026-08-23; since Phase 2 —
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

For the lab machine specifically (one-time setup, dataset transfer, ablation
batch launch/monitoring, results collection), see
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
make smoke            # true end-to-end micro-dataset run (trainer.py, CPU) + fast tests

# Example run (unchanged CLI, now routed through dalmax.cli):
poetry run python trainer.py --dir_results results/dalmax1/ --params_json files_config/benchmark/params_df_gpu_0.json \
    --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda

# Example run using the new generic RepresentationStrategy + embedding/selection config
# (see .specs/experiments/ablation-study.md for the exact params JSON syntax):
poetry run python trainer.py --dir_results results/ablation/ --params_json params_ablation_6_1_spatial.json \
    --dataset_name DANINHAS --strategy_name RepresentationStrategy \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda

# Inference tools (new 2026-08-23, ADR 0006), consuming a trainer.py-written
# saved_model.pth (dalmax-checkpoint format, dalmax/models/checkpoint.py):
poetry run python loader.py --model results/dalmax1/.../saved_model.pth       # inspect a checkpoint
poetry run python predict.py --model results/dalmax1/.../saved_model.pth \
    --dir DATA/daninhas_full/test/DATASET_GRAMINEA --out results/predictions/ # batch prediction
poetry run python predict.py --model results/dalmax1/.../saved_model.pth \
    --image path/to/one_image.jpg                                            # single-image prediction
poetry run python gui.py                                                     # tkinter mini-app

# Lab-machine targets (see LAB_RUNBOOK.md for the full operator guide):
make lab-setup                        # poetry install + torch/CUDA visibility check
make lab-check GPU=0                  # one short real-data GPU run (n_round=1) before a full batch
make ablations-gpu0                   # this GPU's half of the Phase 3 ablation batch (5 configs)
make ablations-gpu1                   # this GPU's half of the Phase 3 ablation batch (6 configs)
make ablation-report                  # aggregate results/ablations/ -> docs/results/ablation_tables/
make benchmark-gpu0 / benchmark-gpu1  # re-run the reference RNHAL sweep (scripts/benchmark/run_pipe_gpu_*.sh)

# Colab targets (see COLAB_RUNBOOK.md for the full notebook-cell guide):
make colab-setup                      # verify Drive mount, build/reuse dataset zip, symlink results/ -> Drive
make colab-check                      # one short real-data GPU run into results/colab_check/ (GPU 0)
make ablations-colab                  # both ablation-script halves sequentially on Colab's single GPU (SKIP_EXISTING=1 default makes relaunch after a disconnect safe)
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
- **Phase 3 — Ablations**: config/code prerequisites all met and materialized
  (`.specs/experiments/ablation-study.md`); **still outstanding**: run the three
  sub-studies on the lab machine and record macro-F1 numbers in
  `ablation-study.md`/`baseline-results.md`.
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
  (`predict.py`/`loader.py`/`gui.py` at the repo root); renamed the historical `demo.py` → `trainer.py`
  (pure rename, `git mv`, no behavior change).

Next milestone: run the three Phase 3 ablation sub-studies on the lab machine and
record their macro-F1 numbers in `ablation-study.md`/`baseline-results.md`, and do
a lab-machine smoke run confirming Phase 4's move didn't break anything there.

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
  `VCTexKmeansHCSampling`, **`RepresentationStrategy`** (NEW — generic, driven
  by the params JSON's `"embedding"`/`"selection"` blocks; see
  `.specs/experiments/ablation-study.md` for exact syntax).
- **Ablation scripts are now idempotent (2026-08-25, `docs/colab-runbook`)**:
  `scripts/ablations/run_ablation_gpu_{0,1}.sh` read `GPU_NUMBER` (default 0/1),
  `SKIP_EXISTING` (default `1` — skips a `(study, config, seed)` triple whose
  `results.json` already exists instead of re-running it), `DRY_RUN` (default
  `0` — echoes commands instead of executing them), and `RESULTS_ROOT`
  (default `results/ablations`) from the environment; the `ExperimentNotifier`
  call is now guarded by `[ -f ExperimentNotifier/main.py ]` (absent on
  Colab). This is what makes `scripts/colab/run_ablations_colab.sh` (both
  scripts pinned to `GPU_NUMBER=0`) safe to relaunch after a Colab disconnect.
  See `tests/test_ablation_scripts.py` and `COLAB_RUNBOOK.md`.
- One params JSON per lab GPU: `files_config/benchmark/params_df_gpu_0.json`,
  `files_config/benchmark/params_df_gpu_1.json`, run
  via `scripts/benchmark/run_pipe_gpu_0.sh` / `scripts/benchmark/run_pipe_gpu_1.sh` (`QUERIES=(10 50 100)`,
  `SEEDS=(1 2 3)`, `n_round 8`, results into `results/dalmax1/`) — unchanged, still
  works via the legacy `config_kmh` key (Phase 2's loader reads it as
  `selection = {method: "hierarchical", hierarchy: config_kmh}` automatically).
