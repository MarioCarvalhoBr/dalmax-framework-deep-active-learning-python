# ADR 0011: TexHAL canonical configuration Q=17

- **Status:** Accepted (author decision 2026-11-01; executed 2026-11-01, verified 2026-11-02; advisor review pending)
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
and the thesis, replacing `Q=[5,17]` (accepted by the author on 2026-11-01; the advisor's review is still pending). The
TexHAL hierarchy and stage ablations are re-run with `Q=17` (see Implementation); the executed `Q=[5,17]`
numbers stay as history, labelled as such.

## Implementation (2026-11-01)

The re-run is implemented as campaign part `texhal_q17` (18 runs = 6 groups x 3 seeds: 6.2 L1/L2a/L2b/L3/L4 and
6.3 w/o hierarchical, `files_config/ablations/texhal_q17/`, VCTex `q=[17]`), not the 24 runs first proposed:
the reference hierarchy with Q=17 and the "full" stage are the already-executed `texhal/6_1/rep_q17` (aliases),
and "w/o representation" is the shared KMH run. Totals become 210 runs / 70 groups. Paper 2's 6.2/6.3 tables
use the new groups, the paper-2/3 comparison tables use `texhal/6_1/rep_q17`, and the executed Q=[5,17]
6.2/6.3 stay as tables `6_2_q5_17`/`6_3_q5_17` (nothing deleted). The runs were executed on 2026-11-01
(Google Colab Pro, about 70 min, 18/18 OK, commit `442a8cd`) and verified on 2026-11-02 (`make campaign-verify`:
210/210 OK, seed audit PASS for seeds 1-3).

## Consequences

- Paper 2 narrative: the headline TexHAL number is 0.8745 weighted F1 (Q=17) instead of 0.8259;
  the 6.1 table is read as "which Q" rather than "single vs. multi-scale".
- Paper 3's comparison table (RNHAL vs TexHAL) uses the new canonical TexHAL run.
- **Outcome (executed 2026-11-01, nq100, weighted F1 / macro F1, mean of seeds 1-3)**: 6.2 best row is
  `L=3 [300,100,50]` 0.8842 / 0.8398, +1.0 point over the reference `[600,200,100]` 0.8745 / 0.8262
  (within one std); 6.3: full 0.8745, w/o hierarchical 0.8454 (-2.9 points), w/o representation 0.8224
  (-5.2 points), so both ablations now hurt and the representation module matters most. TexHAL (Q=17)
  edges the best paper-1 method (LeastConfidence 0.8716, +0.3 points, within std) and stays below RNHAL
  (0.8805, -0.6 points). Details in `.specs/experiments/campaign.md`.
- The old `Q=[5,17]` 6.2/6.3 results are kept as history (append-only), reported as tables
  `6_2_q5_17`/`6_3_q5_17`.
- If the advisor rejects the change, this ADR is marked Rejected/Superseded and the paper-2 tables
  would be reverted to `Q=[5,17]` (the runs exist for both).
- Defaults in `dalmax/config/loader.py` and the VCTex presets are **not** changed by this ADR (no code
  change).
- KI-40 in `.specs/quality/known-issues.md` is resolved (advisor confirmation pending).
