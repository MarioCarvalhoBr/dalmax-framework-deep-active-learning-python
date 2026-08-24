# Git workflow

## Commit style — Conventional Commits

Use conventional commit prefixes:

- `feat:` — new capability (e.g. a new query strategy, a new embedding provider).
- `fix:` — bug fix (e.g. threading the experiment seed into `KMeansSampling`
  instead of the hardcoded `random_state=3`).
- `refactor:` — behavior-preserving restructuring (e.g. replacing an if/elif
  registry with a dict-based one, as done in `dalmax/query_strategies/registry.py`,
  superseding the now-deleted `utils/orchestrator.py`).
- `docs:` — README, CLAUDE.md, `.specs/` changes with no code behavior change.
- `exp:` — experiment configs or results metadata (e.g. a new
  `files_config/benchmark/params_df_gpu_*.json`, an ablation run script, a `results/` metadata update —
  never the raw result artifacts themselves, see `data-safety.md`).

## Commit hygiene

- **Small, focused commits.** One logical change per commit — do not bundle a new
  strategy, a spec update, and an unrelated lint fix into one commit.
- Reference the matching `.specs/` update in the same commit or the immediately
  following one, per `spec-sync.md`.

## Push and branching

- **Never push without explicit user confirmation.** Committing locally is fine
  within a task; `git push` requires the user to say so.
- **Branch for anything touching core `dalmax/` behavior.** Changes to
  `dalmax/query_strategies/`, `dalmax/models/base.py`, `dalmax/models/daninhas_resnet50.py`,
  or `dalmax/tools/` (SSRAE, SSL) should happen on a feature branch, not directly
  on `main` — this repo's `main` is also what gets pulled onto the lab machine
  (see `.claude/commands/handoff-lab.md`), so keeping it stable matters.
- Root-level docs, `.specs/`, and `.claude/` changes may go directly on `main`
  when they are the sole content of the task.

## Safety

- Never run destructive git operations (`reset --hard`, `push --force`,
  `checkout -- .`, `clean -f`, `branch -D`) without explicit user instruction.
- Never skip hooks (`--no-verify`) or bypass signing.
