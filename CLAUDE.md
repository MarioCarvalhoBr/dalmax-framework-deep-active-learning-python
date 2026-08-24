# CLAUDE.md

Operational guide for Claude Code sessions in this repository.

## What this project is

DalMax is a PhD research lab (UFMS) for **Deep Active Learning applied to UAV
weed recognition**. The main contribution is **RNHAL**: a randomized-network
spatio-spectral representation (SSRAE) plus hierarchical k-means batch selection.
Entry point: `demo.py` (a thin shim calling `dalmax.cli.main()`, since Phase 2 —
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

```bash
poetry install       # or: make setup — Poetry-only; the pip/requirements.txt fallback was retired
make lint             # ruff check
make format           # ruff format
make test             # pytest, fast tests only
make smoke            # true end-to-end micro-dataset run (demo.py, CPU) + fast tests

# Example run (unchanged CLI, now routed through dalmax.cli):
poetry run python demo.py --dir_results results/dalmax1/ --params_json files_config/benchmark/params_df_gpu_0.json \
    --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda

# Example run using the new generic RepresentationStrategy + embedding/selection config
# (see .specs/experiments/ablation-study.md for the exact params JSON syntax):
poetry run python demo.py --dir_results results/ablation/ --params_json params_ablation_6_1_spatial.json \
    --dataset_name DANINHAS --strategy_name RepresentationStrategy \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1 --device cuda
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
  metrics, `run_metadata.json`. `demo.py` now routes through `dalmax.cli.main()`.
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
- `demo.py --strategy_name` choices (unchanged plus one new generic name):
  `RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`,
  `SSRAEKmeansSampling`, `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`,
  `VCTexKmeansHCSampling`, **`RepresentationStrategy`** (NEW — generic, driven
  by the params JSON's `"embedding"`/`"selection"` blocks; see
  `.specs/experiments/ablation-study.md` for exact syntax).
- One params JSON per lab GPU: `files_config/benchmark/params_df_gpu_0.json`,
  `files_config/benchmark/params_df_gpu_1.json`, run
  via `scripts/benchmark/run_pipe_gpu_0.sh` / `scripts/benchmark/run_pipe_gpu_1.sh` (`QUERIES=(10 50 100)`,
  `SEEDS=(1 2 3)`, `n_round 8`, results into `results/dalmax1/`) — unchanged, still
  works via the legacy `config_kmh` key (Phase 2's loader reads it as
  `selection = {method: "hierarchical", hierarchy: config_kmh}` automatically).
