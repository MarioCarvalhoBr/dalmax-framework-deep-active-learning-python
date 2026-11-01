# ADR 0011: TexHAL canonical configuration Q=17

- **Status:** Proposed (decided by the user on 2026-11-01; pending advisor confirmation)
- **Date:** 2026-11-01

## Context

The paper-2 method TexHAL (VCTex descriptor + hierarchical k-means) was defined with the canonical
VCTex setting `Q=(5,17)` (two RAEs, 648-d descriptor; `dalmax/config/loader.py` default for `vctex`,
`.claude/rules/reproducibility.md`). The campaign (192 runs, executed 2026-09-30 to 2026-10-05,
`.specs/experiments/campaign.md`) measured at `n_query=100` (final round, mean of seeds 1-3,
weighted F1 / macro F1): `Q=5` 0.8382 / 0.7843, `Q=13` 0.8557 / 0.8052, `Q=17` **0.8745 / 0.8262**
(std 0.0079), `Q=[5,17]` 0.8259 / 0.7687. The canonical setting is the worst of the four and is below
the best paper-1 method (LeastConfidence 0.8716 / 0.8215), while
`Q=17` alone is slightly above it. The TexHAL hierarchy (6.2) and stage (6.3) ablations and the
`shared/texhal_full` run were all executed with `Q=[5,17]`.

## Decision

We will make **`Q=17` (single RAE, 486-d descriptor)** TexHAL's canonical configuration for paper 2
and the thesis, replacing `Q=[5,17]`, **subject to advisor confirmation**. Until then the executed
campaign numbers are reported as they are, with the 6.2/6.3 tables labelled `Q=[5,17]`. We propose
(not yet approved) re-running the TexHAL hierarchy and stage ablations with `Q=17`: 24 runs, with the
new full run replacing `shared/texhal_full` as the canonical TexHAL run in the manifest.

## Implementation (2026-11-01)

The re-run is implemented as campaign part `texhal_q17` (18 runs = 6 groups x 3 seeds: 6.2 L1/L2a/L2b/L3/L4 and
6.3 w/o hierarchical, `files_config/ablations/texhal_q17/`, VCTex `q=[17]`), not the 24 runs first proposed:
the reference hierarchy with Q=17 and the "full" stage are the already-executed `texhal/6_1/rep_q17` (aliases),
and "w/o representation" is the shared KMH run. Totals become 210 runs / 70 groups. Paper 2's 6.2/6.3 tables
use the new groups, the paper-2/3 comparison tables use `texhal/6_1/rep_q17`, and the executed Q=[5,17]
6.2/6.3 stay as tables `6_2_q5_17`/`6_3_q5_17` (nothing deleted). **Status stays Proposed** until the
advisor confirms; the runs are pending (`make campaign-run PART=texhal_q17`).

## Consequences

- Paper 2 narrative: the headline TexHAL number becomes 0.8745 weighted F1 (Q=17) instead of 0.8259;
  the 6.1 table is then read as "which Q" rather than "single vs. multi-scale".
- Paper 3's comparison table (RNHAL vs TexHAL) must use the new canonical TexHAL run.
- If the re-run is approved: manifest regeneration (`scripts/campaign/build_manifest.py`), dedup-table
  and `used_by` updates, `campaign.md`/`ablation-study-texhal.md` update, 24 new GPU runs, and a
  reconciliation of the old `Q=[5,17]` 6.2/6.3 results (kept as history, append-only).
- If the advisor rejects the change, this ADR is marked Rejected and `Q=[5,17]` stays canonical.
- Defaults in `dalmax/config/loader.py` and the VCTex presets are **not** changed by this ADR (no code
  change yet).
- Open: KI-40 in `.specs/quality/known-issues.md`.
