# ADR 0002: Keep the DalMax name; consolidate `core/` + `utils/` into `dalmax/` later

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

The project has been developed under the name "DalMax" (Deep Active Learning Laboratory for UAV
Weed Recognition). During this professionalization pass the question of renaming the project was
raised; the decision was made to **not** rename it — DalMax stays the single project name (not a
"legacy" alias). Separately, the codebase today splits its Python source across two top-level
packages, `core/` (models, strategies, third-party tool wrappers under `core/tools/`) and `utils/`
(dataset loading, orchestration registries, logging, reporting scripts) — see
`.specs/architecture/current-state.md` §1 for the full module map. This split is not driven by any
architectural boundary; it is historical.

`.specs/architecture/target-architecture.md` proposes a single `dalmax/` package (`config/`,
`data/`, `embeddings/`, `selection/`, `query_strategies/`, `models/`, `experiment/`, `reporting/`,
`tools/`) as the end state, consolidating `core/` and `utils/` and adding the new modules required
by the ablation study (`.specs/experiments/ablation-study.md`).

## Decision

1. We will keep **DalMax** as the one and only project name, used in `README.md`, `pyproject.toml`
   (`name = "dalmax"`), and all documentation. No "legacy name" framing is needed since nothing is
   being renamed away from.
2. We will consolidate `core/` and `utils/` into a single `dalmax/` package, but **defer the
   physical move to Phase 4** of `.specs/architecture/refactor-plan.md` ("Polish"), after the core
   abstractions (config layer, embedding provider, selection module, registries — Phase 2) and the
   ablation studies (Phase 3) have landed and been run on the lab machine. Moving files before then
   would add churn risk to already-scheduled lab-machine runs for no immediate benefit.

## Consequences

- Positive: no branding/citation churn — existing citation blocks, paper drafts, and lab-machine
  muscle memory (`git pull` + `run_pipe_gpu_*.sh`) stay valid throughout the refactor.
- Positive: deferring the package move to Phase 4 means Phases 2-3 (the parts that actually change
  runtime behavior and enable the ablations) are not entangled with a large, purely mechanical
  rename — easier to review, easier to bisect if something breaks on the lab machine.
- Negative: until Phase 4, new Phase 2/3 modules live under a provisional location (either still
  under `core/`/`utils/`, or a fresh `dalmax/` package populated ahead of the full rename — the
  exact choice is left to the `implementer` agent executing Phase 2 and must be recorded as a note
  on this ADR or a follow-up ADR when decided).
- Negative: two more phases must pass before the codebase matches `target-architecture.md` exactly;
  `known-issues.md` should track this ADR's reference so nobody mistakes the current `core/`/`utils/`
  split for a stale design.
