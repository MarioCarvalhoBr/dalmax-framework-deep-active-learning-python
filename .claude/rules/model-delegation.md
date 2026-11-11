# Model delegation (quality-first, revised 2026-11-10)

The main session runs **Claude Fable 5.2** and is the orchestrator: it plans, decides, briefs,
reviews and integrates. It delegates the execution to subagents through the Agent tool. The
priority of this policy is **quality of the result**, not token cost: when two tiers could do a job,
pick the stronger one, and spend the savings of the cheap tiers only where judgment is not needed.

This is a standing user instruction: it counts as the user having asked for subagent use.

## Models available (Anthropic first-party prices, USD per million tokens, read 2026-11-10)

| Model | Agent `model` value | Input | Output | Context / max output | Role here |
|---|---|---|---|---|---|
| Claude Fable 5.2 | `fable` (main session) | 8.00 | 40.00 | 1M / 128K | Orchestrator; hardest reasoning |
| Claude Opus 5.5 | `opus` | 4.00 | 20.00 | 1M / 128K | **Default worker** for anything that needs judgment |
| Claude Sonnet 5.5 | `sonnet` | 2.00 | 10.00 | 1M / 128K | Well-specified, bounded execution; wide reading |
| Claude Haiku 5.5 | `haiku` | 0.10 | 0.50 | 1M / 128K | Purely mechanical work with a checkable result |

Reading of the table: Opus 5.5 costs half of Fable 5.2 and is the strongest model below it, so it
is the workhorse. Sonnet 5.5 is half of Opus again. Haiku 5.5 is about forty times cheaper than
Opus, which makes it the right tool for volume, never for decisions. Prices change: re-read them
before relying on these numbers for a budget.

## Who does what

**Fable 5.2, main session (do not delegate):**
- scientific reading of results, claims and their limits, decisions that go to
  `thesis_drafts/PENDENCIAS.md`, ADR-level architecture, experiment design;
- the brief of every delegation and the review of every result;
- subtle debugging once a subagent has failed at it, and conflicts between sources of truth;
- small edits whose delegation would cost more than doing them (a few lines, mid-conversation).

**Opus 5.5 (`opus`), the default for delegated work:**
- implementation in `dalmax/` (features, refactors, bug fixes), tests included;
- code review, experiment audit before any Colab or lab hand-off, security-relevant checks;
- thesis and paper prose: drafting, revising, consistency passes across chapters, related work;
- analysis code of the thesis (`files/latex/analysis/`), table and figure builders;
- reference verification and anything where a wrong answer looks plausible.

**Sonnet 5.5 (`sonnet`):**
- execution of a change that the orchestrator has already specified line by line;
- spec and documentation sync after a change (`.specs/`, runbooks, README) from a list of facts;
- wide read-only sweeps (searching many files, inventories, acronym or number audits) whose
  findings the orchestrator verifies before acting.

**Haiku 5.5 (`haiku`):**
- renames, formatting, running `make lint` / `make test` and reporting, regenerating artefacts,
  copying files, counting; nothing that requires choosing between alternatives.

Rule of thumb when in doubt: **go one tier up**. A redo costs more than the price difference, and
a subagent's mistake arrives in the same confident tone as its findings.

## Effort

Pass `effort` only when the user or a rule asks for it. Otherwise the agent definition decides.
Guidance for agent definitions and explicit requests: review, audit and prose at `high`;
implementation at `high` (`xhigh` for changes that touch seeds, caches or the selection code);
mechanical work at `low`.

## How to delegate well

1. Brief the subagent as a peer: goal, what is already known or ruled out, the exact files, the
   rules that apply (`.claude/rules/`, spec-sync, data safety, no GPU model names, English-only
   code, no em dashes and the self-plagiarism policy in thesis text), and the expected report.
2. Keep the scope narrow and say what must not be touched (`DATA/`, `results/`, `phd_files/`).
3. Independent pieces go out in parallel in one message; dependent pieces go in sequence.
4. Ask for evidence in the report (commands run and their output, file and line), not conclusions.
5. For a review that must not be anchored on the orchestrator's opinion, give the code or the
   text, not the expected verdict.

## Verification (the orchestrator's duty, never delegated away)

- Read the diff of every delegated change before committing it.
- Re-run the gates yourself: `make lint` and `make test` in DalMax; `make check` and `make pdf`
  in the thesis. A subagent saying "tests pass" is not the gate.
- Numbers quoted in prose are checked against `tables/numbers.json` or the source `results.json`.
- Two-stage pattern for work that matters (implementation, experiment configs, thesis sections):
  one agent produces, a **different** Opus 5.5 agent reviews without having seen the first
  agent's reasoning, and the orchestrator arbitrates.

## Agent definitions

`.claude/agents/*.md` carry the tier in their frontmatter and follow this table:

| Agent | Model | Why |
|---|---|---|
| `implementer` | `opus` | implementation needs judgment about seeds, caches and registries |
| `code-reviewer` | `opus` | a missed defect is the most expensive failure |
| `experiment-auditor` | `opus` | gates GPU hours and the validity of results |
| `paper-liaison` | `opus` | prose and code-to-paper consistency |
| `spec-keeper` | `sonnet` | documentation sync from established facts |
| `mechanic` | `haiku` | mechanical, checkable work |

A call may override the tier upward with the Agent tool's `model` argument; never downward for
the first four agents.
