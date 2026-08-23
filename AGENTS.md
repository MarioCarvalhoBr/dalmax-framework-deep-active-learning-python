# AGENTS.md

Tool-agnostic mirror of [`CLAUDE.md`](CLAUDE.md) for non-Claude agentic tools.
For full detail, always defer to `.specs/` and `.claude/`.

## Project

DalMax — a PhD research lab (UFMS) for Deep Active Learning applied to UAV weed
recognition. Main contribution: **RNHAL**, combining a randomized-network
spatio-spectral representation (SSRAE) with hierarchical k-means batch selection.
Entry point: `demo.py` (a thin shim calling `dalmax.cli.main()`). All Python
source lives in one package, `dalmax/` — the old `core/`/`utils/` split no
longer exists (consolidated in Phase 4).

## Source of truth

- [`.specs/00-overview.md`](.specs/00-overview.md) — project summary, method, status.
- [`.specs/README.md`](.specs/README.md) — index of all specs, spec-sync contract.
- [`.claude/agents/`](.claude/agents/) — role definitions (code-reviewer,
  implementer, mechanic, experiment-auditor, spec-keeper, paper-liaison); use
  these as a guide for how work should be divided even outside Claude Code.

## Setup & commands

```bash
poetry install     # or: make setup
make lint            # ruff check
make format          # ruff format
make test            # pytest, fast tests only
make smoke           # true end-to-end micro-dataset run (demo.py, CPU) + fast tests
make export-reqs     # regenerate requirements.txt from pyproject.toml
```

## Working mode

Prefer delegating routine implementation to a lower-capability/cheaper model and
reserving the strongest model for genuinely complex design/debugging decisions —
see [`.claude/rules/model-delegation.md`](.claude/rules/model-delegation.md).
Any change to behavior, CLI, params schema, strategy registry, or experiment
protocol must update the matching `.specs/` file in the same change (spec-sync,
[`.claude/rules/spec-sync.md`](.claude/rules/spec-sync.md)); new architectural
decisions get an ADR under `.specs/adr/`.

## Non-negotiables

- Code quality: [`.claude/rules/code-quality.md`](.claude/rules/code-quality.md)
- Reproducibility: [`.claude/rules/reproducibility.md`](.claude/rules/reproducibility.md)
- Data safety: [`.claude/rules/data-safety.md`](.claude/rules/data-safety.md)
- Git workflow: [`.claude/rules/git-workflow.md`](.claude/rules/git-workflow.md)

## Current phase

Phase 1 (safety net: tests, golden run, CI — done) → Phase 2 (core refactor:
config layer, embedding provider abstraction, registries — done) → Phase 3
(ablation study: config/code done, lab-machine runs outstanding) → Phase 4
(polish: package consolidated into `dalmax/`, dead code deleted — done). Details:
[`.specs/architecture/refactor-plan.md`](.specs/architecture/refactor-plan.md).

## Never do

- Write into `DATA/` or edit/delete anything under `results/`.
- Run full training locally (no GPU on the dev machine); use `make smoke` for
  local CPU checks only.
- Commit `.pkl`, `.pth`, or other large/generated artifacts.
- Push to the remote without explicit user confirmation.

## Key facts

- SSRAE `Q = 13`; VCTex `Q ∈ {5, 17}`.
- Results dir convention: `{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/`.
- `demo.py --strategy_name` choices: `RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`,
  `SSRAEKmeansSampling`, `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`,
  `VCTexKmeansHCSampling`, plus the generic `RepresentationStrategy` (driven by
  the params JSON's `"embedding"`/`"selection"` blocks).
- `--device {auto,cuda,cpu}` and `--embedding_variant {full,spatial,spectral}`
  are additional CLI flags on top of the original set.
- One params JSON per lab GPU (`params_df_gpu_0.json`, `params_df_gpu_1.json`),
  run via `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`.
