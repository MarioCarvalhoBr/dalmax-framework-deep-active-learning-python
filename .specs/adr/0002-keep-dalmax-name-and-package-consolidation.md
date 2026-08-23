# ADR 0002: Keep the DalMax name; consolidate `core/` + `utils/` into `dalmax/` later

- **Status:** Accepted (amended 2026-08-23 — see "Amendment" below)
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

## Amendment (2026-08-23, Phase 2 landed)

The "exact choice ... left to the `implementer` agent executing Phase 2" (Consequences, above) was
resolved: **the `dalmax/` package was created immediately in Phase 2**, not staged under
`core/`/`utils/` first. Every genuinely new Phase 2 abstraction (config layer, embedding providers +
cache, selection strategies, the generic `RepresentationStrategy`, all registries, the experiment
runner/reporter/run-metadata modules, `seeding.py`) lives under `dalmax/` as of commit `ed37f8a`
(branch `refactor/phase-2-core`). `core/` and `utils/` were **not** moved or deleted — every legacy
model class, the training engine (`core/deep_learning.py`), dataset loaders (`utils/data.py`,
`utils/dataset.py`), and the 12 non-representation strategy classes are called by `dalmax/` code
(`dalmax/models/registry.py`, `dalmax/data/registry.py`,
`dalmax/query_strategies/registry.py::LEGACY_STRATEGY_REGISTRY`) exactly as they are, unmodified.
Four legacy files became **dead code** rather than being deleted (`utils/orchestrator.py`,
`core/query_strategies/{ssrae_kmeans_sampling,vctex_kmeans_sampling,ssl_ssrae_sampling}.py`) — kept
in place per this ADR's original "defer the physical move to Phase 4" decision, scheduled for
deletion there (see `.specs/architecture/refactor-plan.md` Phase 4 and
`.specs/architecture/current-state.md` §0 for exactly why each is unreachable from `demo.py`/
`dalmax.cli` today). This amendment does not change the original Decision (DalMax keeps its name;
full consolidation is still Phase 4) — it only records which of the two staging options was taken.
