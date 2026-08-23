# Model delegation for coding tasks

For all coding tasks, the main session (running a high-power model, e.g. Fable) acts
strictly as an orchestrator — it delegates, provides context, and reviews — and
participates directly only when a case is genuinely complex enough to need the top
model. Use judgement to pick an appropriate lower-power model and run the actual
coding in a subagent via the Agent tool:

- **`sonnet`** — routine implementation work: new features, refactors, survey JSON
  authoring, bug fixes with a known cause.
- **`haiku`** — trivial/mechanical changes: renames, copy tweaks, small config edits,
  running validations.
- **Main session directly** — only when delegation would clearly cost more than it
  saves (one-line edits mid-conversation) or when the task genuinely needs the top
  model (subtle debugging, architecture decisions).

Give the subagent full context in its prompt (relevant files, conventions from
`.claude/rules/`, the spec-sync requirement), then review its output in the main
session before considering the task done.

This is a standing user instruction: it counts as the user having asked for subagent use.
