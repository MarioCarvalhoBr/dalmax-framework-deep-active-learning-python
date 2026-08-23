# CLAUDE.md

Operational guide for Claude Code sessions in this repository.

## What this project is

DalMax is a PhD research lab (UFMS) for **Deep Active Learning applied to UAV
weed recognition**. The main contribution is **RNHAL**: a randomized-network
spatio-spectral representation (SSRAE) plus hierarchical k-means batch selection.
Entry point: `demo.py`, orchestrated via `utils/orchestrator.py`.

## Source of truth

Specs, not this file, are authoritative for architecture, protocol, and status:

- [`.specs/00-overview.md`](.specs/00-overview.md) — what DalMax is, method summary, current status.
- [`.specs/README.md`](.specs/README.md) — index of all specs and the spec-sync contract.

If code, this file, and `.specs/` disagree, treat `.specs/` as correct and fix the
drift (see spec-sync rule below) rather than trusting stale prose here.

## Setup & commands

```bash
poetry install       # or: make setup
make lint             # ruff check
make format           # ruff format
make test             # pytest, fast tests only
make smoke            # import checks + fast tests (true micro-dataset run = Phase 1 deliverable)
make export-reqs      # regenerate requirements.txt from pyproject.toml

# Example run:
python demo.py --dir_results results/dalmax1/ --params_json params_df_gpu_0.json \
    --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
    --n_query 100 --n_init_labeled 100 --n_round 8 --seed 1
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

- **Phase 1 — Safety net** (current): tests, golden-run capture, CI green.
- **Phase 2 — Core refactor**: config layer, embedding provider abstraction
  (SSRAE/VCTex/ResNet-ImageNet) with keyed cache, selection module abstraction,
  strategy/dataset/model registries, seed-propagation audit.
- **Phase 3 — Ablations**: implement the three studies in
  [`.specs/experiments/ablation-study.md`](.specs/experiments/ablation-study.md)
  as configs, run on the lab machine, generate the paper's ablation tables.
- **Phase 4 — Polish**: package rename to `dalmax/`, docs refresh, dead-code removal.

Next milestone: close out Phase 1 (tests + CI green), then start Phase 2.

## Never do

- Touch or write into `DATA/` (immutable input).
- Edit or delete anything under `results/` (append-only experiment history).
- Run a full training/experiment locally (no GPU here) — local runs are for
  `make smoke` / tiny CPU subsets only; real training happens on the lab machine
  or Colab.
- Commit `.pkl`, `.pth`, or other large/generated artifacts.
- `git push` without explicit user confirmation.

## Key facts cheat-sheet

- SSRAE hidden-layer size `Q = 13` (`utils/data.py`, `create_feature_maps_ssrae`);
  VCTex uses `Q ∈ {5, 17}`.
- Results dir: `{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/`.
- `demo.py --strategy_name` choices: `RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`,
  `SSRAEKmeansSampling`, `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`,
  `VCTexKmeansHCSampling`.
- One params JSON per lab GPU: `params_df_gpu_0.json`, `params_df_gpu_1.json`, run
  via `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh` (`QUERIES=(10 50 100)`,
  `SEEDS=(1 2 3)`, `n_round 8`, results into `results/dalmax1/`).
