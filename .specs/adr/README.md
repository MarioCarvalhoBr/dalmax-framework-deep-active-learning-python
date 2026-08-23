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
| [0002](0002-keep-dalmax-name-and-package-consolidation.md) | Keep the DalMax name; consolidate `core/` + `utils/` into `dalmax/` later | Accepted |
| [0003](0003-embedding-provider-abstraction.md) | Introduce an `EmbeddingProvider` abstraction with a keyed cache | Accepted |
| [0004](0004-micro-dataset-and-golden-run.md) | Deterministic micro-dataset + golden-run fixtures for local smoke testing | Accepted |

## When to add a new ADR

- A decision changes the public shape of the codebase (package layout, registry contracts, config
  schema, CLI surface) — not a routine bug fix or refactor that follows an existing ADR.
- A decision trades off between two or more genuinely viable options and future readers will want
  to know why the chosen one won.
- A decision reverses or narrows an earlier ADR — write a new ADR and mark the old one's `Status`
  as `Superseded by ADR-NNNN` rather than editing history.
