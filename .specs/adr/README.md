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

## When to add a new ADR

- A decision changes the public shape of the codebase (package layout, registry contracts, config
  schema, CLI surface) — not a routine bug fix or refactor that follows an existing ADR.
- A decision trades off between two or more genuinely viable options and future readers will want
  to know why the chosen one won.
- A decision reverses or narrows an earlier ADR — write a new ADR and mark the old one's `Status`
  as `Superseded by ADR-NNNN` rather than editing history.
