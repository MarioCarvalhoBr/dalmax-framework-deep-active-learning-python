# Architecture Decision Records — Index

This directory records significant architectural decisions for DalMax as ADRs (one file per
decision, numbered sequentially, never renumbered or deleted after acceptance — superseded
decisions get a new ADR that references the old one).

Use `template.md` as the starting point for any new ADR (`Status` / `Context` / `Decision` /
`Consequences`). Per `.claude/rules/spec-sync.md`, any new architectural decision made during a
session must be captured here in the same task, not deferred.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-adopt-poetry.md) | Adopt Poetry for dependency management | Accepted |
| [0002](0002-keep-dalmax-name-and-package-consolidation.md) | Keep the DalMax name; consolidate `core/` + `utils/` into `dalmax/` later | Accepted; **fully executed** (amended 2026-08-23 twice — Phase 2 `dalmax/` creation, then the Phase 4 physical move/delete that closes the ADR) |
| [0003](0003-embedding-provider-abstraction.md) | Introduce an `EmbeddingProvider` abstraction with a keyed cache | Accepted |
| [0004](0004-micro-dataset-and-golden-run.md) | Deterministic micro-dataset + golden-run fixtures for local smoke testing | Accepted |
| [0005](0005-representation-strategy-and-registries.md) | One generic `RepresentationStrategy` + dict registries replacing the four fixed embedding-based strategy classes and the (now-deleted) `utils/orchestrator.py`'s if/elif chains | Accepted |
| [0006](0006-checkpoint-format-and-inference-tools.md) | Self-describing `dalmax-checkpoint` format fixing the historical save/load bug, plus standalone inference tools (`predict.py`, `loader.py`, `gui.py`) and the `demo.py` → `trainer.py` rename | Accepted |
| [0007](0007-per-method-ablation-layout.md) | Organize ablation configs/scripts/results by method (RNHAL vs. TexHAL) via a `METHOD` variable | Accepted |
| [0008](0008-single-campaign-manifest-dedup.md) | One campaign manifest drives a single, deduplicated re-execution of all three papers (aliases, redundancy checker, `used_by`) | Accepted |
| [0009](0009-shared-no-representation-run.md) | One shared "w/o representation module" run for papers 2 and 3 (and paper 1's KMH@100); supersedes the 2026-09-01 per-paper decision | Accepted |
| [0010](0010-cleanup-tools-and-hardware-agnostic.md) | `tools/` for CLIs, the campaign as the only run path (legacy scripts/reporters removed), hardware-agnostic repository (no GPU model names; GPU recorded at runtime), determinism/resume/seed-audit follow-ups, `--exclude-strategy` | Accepted |
| [0011](0011-texhal-canonical-q17.md) | TexHAL canonical configuration `Q=17` (single RAE) instead of `Q=[5,17]`, from the campaign results; 6.2/6.3 ablations re-run with Q=17 (part `texhal_q17`, executed 2026-11-01) | Accepted (author 2026-11-01; advisor review pending) |
| [0012](0012-kmh-parameter-sweep.md) | KMH parameter sweep at `n_query` 10/50/100 (hierarchy shapes incl. budget-matched ones, k-means init, leaf sampling, resampling), as a separate manifest; four optional `selection.hierarchy` keys with behaviour-preserving defaults | Accepted (author 2026-11-07; executed 2026-11-07/08, 129/135 first pass; 6 init_random runs re-run after the KI-44 fix); stage 2 (nq100 only, 10 groups / 30 runs, prepared 2026-11-08) executed 2026-11-09/10, verified 165/165 on 2026-11-10; nq100 winner `hier_L2_matched__no_resampling` 0.8819 |

## When to add a new ADR

- A decision changes the public shape of the codebase (package layout, registry contracts, config
  schema, CLI surface) — not a routine bug fix or refactor that follows an existing ADR.
- A decision trades off between two or more genuinely viable options and future readers will want
  to know why the chosen one won.
- A decision reverses or narrows an earlier ADR — write a new ADR and mark the old one's `Status`
  as `Superseded by ADR-NNNN` rather than editing history.
