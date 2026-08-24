# Sync between code, `.specs/` and `.claude/`

DalMax is a research lab where the specs are the paper's supporting evidence trail,
not decoration. Specs and rules must never drift from what the code actually does.

## The contract

1. **Any change to behavior, CLI, params schema, strategy registry, or experiment
   protocol MUST update the corresponding `.specs/` file in the same task.**
   Examples of triggers and their target spec files:
   - Adding/removing a `trainer.py --strategy_name` choice, or editing
     `dalmax/query_strategies/registry.py` / `dalmax/query_strategies/__init__.py` →
     update `.specs/use-cases/add-new-strategy.md` and
     `.specs/architecture/current-state.md`.
   - Changing `files_config/benchmark/params_df_gpu_*.json` schema (e.g. `config_kmh`, `n_classes`) →
     update `.specs/experiments/experimental-protocol.md`.
   - Changing results directory naming in `dalmax/experiment/runner.py`
     (`{dir_results}/{dataset}/SEED_{seed}/NQ_{n_query}_NIL_{n_init}_NR_{n_round}_NE_{n_epoch}/{strategy}/`) →
     update `.specs/experiments/experimental-protocol.md` and
     `.specs/research-rules/reproducibility.md`.
   - Changing the embedding cache (`dalmax/embeddings/cache.py::EmbeddingCache`,
     `results/cache/embeddings/*.pkl`) or the SSRAE `Q`
     value (`dalmax/config/loader.py`) → update `.specs/experiments/ablation-study.md`
     (representation ablation section) and `.specs/quality/known-issues.md`.

2. **New architectural decisions get an ADR** in `.specs/adr/`, following
   `.specs/adr/template.md`, and are added to `.specs/adr/README.md`'s index.
   Example: choosing to abstract embedding providers (SSRAE | VCTex |
   ResNet-ImageNet) behind one interface is exactly the kind of decision that
   needs an ADR (see `0003-embedding-provider-abstraction.md`).

3. **Whenever new rules or conventions emerge in conversation, they must be
   persisted** into `.claude/rules/` (for agent-facing operational rules) and
   into the relevant `.specs/` file (for the human-facing record). Do not let a
   convention live only in a chat transcript.

## Practical checklist before closing a task

- [ ] Did this change touch `trainer.py`/`dalmax/cli.py` CLI args, `dalmax/query_strategies/registry.py`, or
      `dalmax/query_strategies/__init__.py`? → update `.specs/use-cases/add-new-strategy.md`.
- [ ] Did this change touch a params JSON schema or the results directory layout?
      → update `.specs/experiments/experimental-protocol.md`.
- [ ] Did this change introduce a new module, provider, or registry?
      → write an ADR.
- [ ] Did this change fix or reveal a known issue?
      → update `.specs/quality/known-issues.md` (mark resolved or add new entry).
- [ ] Did this session establish a new working convention?
      → write it into `.claude/rules/` and reference it from `.specs/README.md`.

See also `.claude/agents/spec-keeper.md`, whose job is to run this checklist
after merged changes.
