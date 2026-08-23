---
description: Run the code-reviewer agent on the current diff.
---

Run the `code-reviewer` agent (`.claude/agents/code-reviewer.md`) against the
current working tree diff.

Steps:

1. Run `git status` and `git diff` (and `git diff --staged` if there are staged
   changes) to gather the full diff. If `$ARGUMENTS` names a different target
   (a commit range, a branch, or specific file paths), use that instead of the
   working tree diff.
2. Invoke the `code-reviewer` agent with the diff as context, asking it to check
   correctness, reproducibility (seed handling, cache invalidation per
   `.claude/rules/reproducibility.md`), performance on the lab's 10 GB GPUs, and
   adherence to `.claude/rules/*.md`.
3. If the diff touches `core/query_strategies/`, `utils/orchestrator.py`, or
   `demo.py`'s strategy choices, explicitly ask the agent to verify registry
   consistency (see `.claude/agents/code-reviewer.md` point 5) and cross-check
   against `.specs/use-cases/add-new-strategy.md`.
4. Relay the agent's findings verbatim, grouped by severity (Blocking /
   Should-fix / Nit).

Do not auto-apply fixes — this command reports only. To act on findings,
delegate to `implementer` or `mechanic` per `.claude/rules/model-delegation.md`.
