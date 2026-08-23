# <Spec title>

Status: Draft | Active | Superseded — last updated YYYY-MM-DD

## Purpose

One paragraph: what question does this spec answer, and who needs the answer (a future
implementer, the advisor, a paper co-author)?

## Scope

What this spec covers and, just as importantly, what it explicitly does not cover (link to the
sibling spec that does, if any).

## Content

The body of the spec. Prefer:
- Tables over prose when describing schemas, file mappings, or comparisons.
- Direct file/line references (`path/to/file.py:NN`) for any claim about current code behavior —
  verify before writing, never guess.
- `TBD` (with a one-line note on what's missing and who/what would resolve it) instead of an
  invented fact.
- A Mermaid diagram when a flow, hierarchy, or set of relationships is easier to see than to read.

## Open questions

Anything genuinely undecided, phrased as a question, so a future session can pick it up without
re-deriving the context.

## Related specs

- Link to other `.specs/` files this one depends on or is depended on by.
- Link to the ADR(s) that formalized any decision referenced here.

---
Per `.claude/rules/spec-sync.md`: any code change that affects the claims in this spec must update
this file in the same task. If you're reading this template to start a new spec, delete this
footer and everything above "## Purpose" is yours to fill in.
