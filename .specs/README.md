# .specs — index

This folder is the source of truth about the DalMax research codebase: what
exists today, what the experiments mean, and what is planned. It exists
because Claude Code sessions (and humans returning after months away from a
PhD codebase) need a place to recover context without re-reading every
script.

## Why `experiments/`, `research-rules/`, `adr/` instead of `api/`, `frontend/`, `business-rules/`

The classic spec skeleton used for product/web-app repos (`api/`,
`frontend/`, `business-rules/`) does not fit a research laboratory: there is
no API surface or frontend, and the invariants that matter are not "business
rules" but **experimental protocol** (seeds, splits, budgets — violating
them invalidates a comparison, not a feature), **reproducibility rules**
(what must be pinned/logged for a run to be redoable), and **architecture
decisions** (ADRs) about the codebase supporting that research. Hence:

- `experiments/` replaces `business-rules/` — the protocol that governs
  what a "valid experiment" is (datasets, splits, seeds, budgets, metrics,
  results layout) and the ablation study specification.
- `research-rules/` replaces the narrower "business rules" framing —
  reproducibility, metric definitions, and dataset handling rules that must
  hold for any result to be citable in the paper.
- `adr/` (Architecture Decision Records) is kept as-is; it plays the same
  role it would in a product repo.
- There is no `api/` or `frontend/` because none exist in this repo.

## Full index

```
.specs/
├── 00-overview.md                        # what DalMax is, RNHAL summary, status
├── README.md                             # this file
├── architecture/
│   ├── current-state.md                  # [owned elsewhere]
│   ├── target-architecture.md            # [owned elsewhere]
│   └── refactor-plan.md                  # [owned elsewhere]
├── adr/
│   ├── README.md                         # [owned elsewhere]
│   ├── 0001-adopt-poetry.md              # [owned elsewhere]
│   ├── 0002-keep-dalmax-name-and-package-consolidation.md  # [owned elsewhere]
│   ├── 0003-embedding-provider-abstraction.md              # [owned elsewhere]
│   ├── 0004-micro-dataset-and-golden-run.md                # [owned elsewhere]
│   ├── 0005-representation-strategy-and-registries.md      # [owned elsewhere]
│   ├── 0008-single-campaign-manifest-dedup.md              # one manifest, aliases, redundancy checker
│   ├── 0009-shared-no-representation-run.md                # one shared "w/o representation" run (supersedes 2026-09-01)
│   ├── 0010-cleanup-tools-and-hardware-agnostic.md         # tools/ CLIs, campaign as the only run path, hardware-agnostic repo
│   └── template.md                       # [owned elsewhere]
├── experiments/
│   ├── experimental-protocol.md          # datasets, splits, seeds, budgets, metrics, results layout
│   ├── ablation-study.md                 # RNHAL (paper 3) ablation spec (representation / hierarchy / stage-contribution) — EXECUTED
│   ├── ablation-study-texhal.md          # TexHAL (paper 2) ablation spec, same structure — materialized, not yet run
│   ├── papers-roadmap.md                 # the three-paper plan (DAL benchmark / TexHAL / RNHAL) and how it maps onto the codebase
│   ├── campaign.md                       # the single deduplicated re-execution of all three papers (manifest, dedup table, counts, commands)
│   └── baseline-results.md               # where existing results/ runs live, what they cover
├── research-rules/
│   ├── reproducibility.md                # seed policy, cache policy, config snapshot per run
│   ├── metrics.md                        # exact metric definitions (sklearn averaging actually used)
│   └── dataset-protocol.md               # daninhas_full layout, per-class counts, imbalance notes
├── infrastructure/
│   ├── execution-environments.md         # local / lab / Colab Pro decision matrix (hardware-agnostic)
│   └── environment-setup.md              # Poetry install, .venv, lab/Colab pip fallback, CUDA notes
├── quality/
│   ├── code-standards.md                 # [owned elsewhere]
│   ├── testing-strategy.md               # [owned elsewhere]
│   └── known-issues.md                   # [owned elsewhere]
├── templates/
│   ├── spec-template.md                  # [owned elsewhere]
│   └── experiment-report-template.md     # [owned elsewhere — TBD confirm exists, not found as of this writing]
├── use-cases/
│   ├── add-new-strategy.md               # every file that must change to add a query strategy
│   ├── run-full-benchmark.md             # all strategies × seeds × budgets on the lab machine
│   ├── run-ablation.md                   # how to execute the three ablation studies
│   └── generate-report.md                # dalmax/reporting/ pipeline, raw results → averaged tables
├── history/
│   └── prompt-master.md                  # the original scaffolding master prompt (historical record; moved 2026-09-29)
└── future/
    └── ideas.md                          # backlog (package rename, config layer, tracking, DVC, ...)
```

Files marked `[owned elsewhere]` are assigned to a different scaffolding
work batch (architecture/, adr/, quality/, templates/) per the master
prompt's Deliverable C split; this batch (C2) is responsible for everything
else listed above, including this index and `00-overview.md`. As of this
writing, `architecture/`, `adr/`, and `quality/` are all present, and
`templates/spec-template.md` exists but
`templates/experiment-report-template.md` does not yet — treat that one
link as forward-looking, not broken.

## Spec-sync contract — when to update what

- **Any change to CLI flags, params-JSON schema, the strategy registry, or
  results directory naming** (in `demo.py` (historical), `dalmax/cli.py`,
  `dalmax/query_strategies/registry.py`, any `dalmax/query_strategies/*`) →
  update `experiments/experimental-protocol.md` and, if a new strategy was
  added, `use-cases/add-new-strategy.md` in the **same task**.
- **Any change to how metrics are computed** (`dalmax/data/datasets.py`'s
  `Data.calc_metrics`/`calc_metrics_sklearn` or equivalent) → update
  `research-rules/metrics.md` in the same task; this affects every number
  that ends up in the paper.
- **Any change to caching** (`dalmax/embeddings/cache.py`'s `EmbeddingCache`)
  → update `research-rules/reproducibility.md` (cache-key policy) in the
  same task.
- **Any new architectural decision** (e.g., adopting an embedding-provider
  abstraction, a config layer) → add an ADR under `adr/` (owned by the other
  batch — flag it there) and cross-reference it from the relevant file here.
- **Any run executed on the lab machine or Colab that produces new
  `results/`** → update `experiments/baseline-results.md` with what the run
  covers (strategy/params/seed range) and where it landed.
- **New rules or conventions that emerge in conversation** (this is a
  standing instruction from the master prompt) must be persisted into both
  `.claude/rules/` (owned by a different batch) and the corresponding file
  here — do not let a convention live only in chat history.
- Whenever you introduce a `TBD` for something that later becomes known,
  replace it in place rather than leaving stale placeholders.
