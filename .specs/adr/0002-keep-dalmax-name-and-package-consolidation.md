# ADR 0002: Keep the DalMax name; consolidate `core/` + `utils/` into `dalmax/` later

- **Status:** Accepted; **fully executed** (amended 2026-08-23 twice — Phase 2 staging note, then the
  Phase 4 consolidation itself — see "Amendments" below)
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
  muscle memory (`git pull` + `scripts/benchmark/run_pipe_gpu_*.sh`) stay valid throughout the refactor.
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

## Amendments

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

### Amendment (2026-08-23, Phase 4 executed — consolidation complete)

Phase 4 landed on branch `refactor/phase-4-package` (`git log --oneline b51fb59..HEAD`: `307c7f1`
query strategies moved, `38c4b0e` models/data/tools/reporting moved, `d58e116` dead code deleted,
`48a4682` `dalmax/tools/` follow-up, `79e628c` vendored `.DS_Store` cleanup). `core/` and `utils/`
**no longer exist on disk** — this is a physical move, not a wrap:

- `core/query_strategies/*.py` (12 baseline classes + `strategy.py`) → `dalmax/query_strategies/*.py`
  (module names kept 1:1, e.g. `strategy.py` → `base.py`; the family-grouping layout originally
  sketched in `target-architecture.md` §2 — `uncertainty.py`/`diversity.py`/`bayesian.py`/
  `adversarial.py` — was **not** adopted; `target-architecture.md` has been amended to document the
  1:1 layout as the chosen final state).
- `core/deep_learning.py` → `dalmax/models/base.py`; `core/daninhas_model.py` →
  `dalmax/models/daninhas_resnet50.py` (the unused `DaninhasModelVitB16` class was dropped, not
  carried over — see `known-issues.md` KI-25, now resolved by deletion); `core/cifar10_model.py` →
  `dalmax/models/cifar10_cnn.py`.
- `core/tools/{SSRAE,VCTex,SSL}/` → `dalmax/tools/{SSRAE,VCTex,SSL}/`, vendored contents otherwise
  unchanged (still Meta-licensed for `SSL/`, wrap-don't-edit policy still applies to file *contents*,
  just not to their location any more).
- `utils/report/*.py` → `dalmax/reporting/*.py`, renumbered to descriptive names
  (`extract_confusion_matrices.py`, `chunk_results.py`, `average_confusion_matrices.py`,
  `average_results.py`, `build_method_metrics.py`, `plot_results_dir.py`) plus the new
  `ablation_report.py`.
- `utils/data.py` → **deleted** (683 lines; not moved — its live logic had already been fully
  superseded by `dalmax/data/{datasets,handlers,loaders,registry}.py` and `dalmax/embeddings/` in
  Phase 2/3). `utils/LOGGER.py` → `dalmax/logging_utils.py` (moved, not deleted — still the one
  active logger).
- `utils/orchestrator.py` (80 lines) and the three superseded strategy files
  (`ssrae_kmeans_sampling.py`, `vctex_kmeans_sampling.py`, `ssl_ssrae_sampling.py`) — **deleted**,
  not carried over, since Phase 2's registries (`dalmax/{data,models,query_strategies}/registry.py`)
  and `dalmax/query_strategies/representation.py` had already fully superseded them and nothing
  reachable imported them (verified before deletion).
- Also deleted as scratch/dead: `demo_ssl.py`, `temp_teste.py`, `test.py`,
  `core/query_strategies/old_functions.py`, `code_kmh.py` (under vendored SSL tools),
  `create_feature_maps_ssrae`/`create_feature_maps_vctex` (legacy-only methods, went with
  `utils/data.py`), `plot_features_tsne_*` (legacy-only), `TRASH_TEXT.md`, `sampled_data.pdf`,
  `TODO.md`, and orphaned `results/*.pkl` files that were local scratch, not `results/` outputs a
  real experiment run produced.
- `demo.py` remains at the repo root as a thin backward-compatible shim; it is no longer where the
  real logic lives (that's `dalmax/cli.py` and `dalmax/experiment/`).

Verification performed before this amendment was written: `ls core utils` → both "No such file or
directory"; `grep -rn "^import core\|^from core\|^import utils\|^from utils"` across `dalmax/`,
`tests/`, `demo.py` → zero hits; all 17 CLI strategy names resolve via
`dalmax/query_strategies/registry.py`; golden-run fixtures bit-identical; 277 tests pass
(`poetry run pytest -q -m "not gpu and not dataset and not slow"`).

This closes the ADR: the Decision in both the original body and the Phase 2 amendment is now fully
realized — DalMax kept its name throughout, and `core/`/`utils/` are consolidated into `dalmax/`
with nothing left behind. `.specs/architecture/refactor-plan.md`'s Phase 4 acceptance criteria and
`.specs/quality/known-issues.md` have been updated to match (see those files for the itemized
before/after and the closed known-issue rows).
