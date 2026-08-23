---
description: Scan recent git changes and update affected .specs/ files, reporting any drift.
---

Scan recent changes and bring `.specs/` back in sync with the code, per the
contract in `.claude/rules/spec-sync.md`.

Steps:

1. Run `git log --oneline -20` and `git diff HEAD~5 --stat` (adjust the range if
   the user specified one in `$ARGUMENTS`) to see what changed recently. If
   `$ARGUMENTS` names a specific commit range, ref, or file, scope the scan to
   that instead of the last 5 commits.
2. For each changed file, map it to the spec(s) it should have updated, using
   the table in `.claude/rules/spec-sync.md`:
   - `demo.py`, `dalmax/cli.py`, `dalmax/query_strategies/registry.py`,
     `dalmax/query_strategies/__init__.py` →
     `.specs/use-cases/add-new-strategy.md`, `.specs/architecture/current-state.md`.
   - `params_df_gpu_*.json`, results directory naming in `dalmax/experiment/runner.py` →
     `.specs/experiments/experimental-protocol.md`.
   - `dalmax/embeddings/cache.py` (embedding cache, `Q` values) →
     `.specs/experiments/ablation-study.md`, `.specs/quality/known-issues.md`.
   - New module/provider/registry → check `.specs/adr/` for a matching or
     missing ADR.
3. For each spec that is now stale, either update it directly (for
   straightforward factual sync — e.g. a strategy was added and the use-case
   doc needs its name added to a list) or delegate to the `spec-keeper` agent
   (`.claude/agents/spec-keeper.md`) for anything requiring judgment, per
   `.claude/rules/model-delegation.md`.
4. Report drift found and fixed, and any drift that still needs a human
   decision (e.g. the code and the paper disagree on notation — that's
   `paper-liaison`'s territory, not this command's).

Output: a short table `spec file | status (in sync / updated / needs human
decision) | note`.
