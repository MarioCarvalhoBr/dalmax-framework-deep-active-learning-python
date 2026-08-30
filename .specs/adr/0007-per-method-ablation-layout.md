# ADR 0007: Organize ablation configs/scripts/results by method (RNHAL vs. TexHAL) via a METHOD variable

- **Status:** Accepted
- **Date:** 2026-08-30

## Context

The codebase committed to a three-paper plan (`.specs/experiments/papers-roadmap.md`): paper 1 is a
DAL benchmark comparison (no representation-based strategies), paper 2 proposes **TexHAL**
(VCTex + hierarchical k-means selection), and paper 3 proposes **RNHAL** (SSRAE + hierarchical
k-means selection, ablations already executed 2026-08-26 on Colab, see
`.specs/experiments/ablation-study.md`). Until this ADR, `files_config/ablations/` held only
RNHAL's 11 configs (+ 11 CPU-smoke micro mirrors), and `scripts/ablations/run_ablation_gpu_{0,1}.sh`
/ `smoke_ablations.sh` / `scripts/colab/run_ablations_colab.sh` / `dalmax/reporting/
ablation_report.py` / the Makefile's `ablations-*`/`ablation-report` targets all hardcoded that
single suite's paths and config basenames.

Adding TexHAL's own ablation suite (12 configs — one more §6.1 row than RNHAL's, since VCTex has no
spatial/spectral split and instead sweeps its own `Q` scale hyperparameter, see
`.specs/experiments/ablation-study-texhal.md`) needed a place to live that would not collide with,
or be confused with, RNHAL's already-executed batch — and the two suites needed to be runnable
independently (a lab session might run only one) without duplicating every script.

## Decision

We will organize ablation configs, results, and reports **by method**, selected by a `METHOD`
environment variable (`rnhal` default, or `texhal`) threaded through every touchpoint:

- **Configs**: `files_config/ablations/{rnhal,texhal}/*.json` (+ each method's own `micro/`
  mirror). RNHAL's existing 11 configs moved from `files_config/ablations/` directly into
  `files_config/ablations/rnhal/` (pure `git mv`, no content change); TexHAL's 12 configs are new.
- **Scripts**: `scripts/ablations/run_ablation_gpu_{0,1}.sh`, `scripts/ablations/
  smoke_ablations.sh`, and `scripts/colab/run_ablations_colab.sh` all read `METHOD` (validated,
  exit 2 on an invalid value) and select `files_config/ablations/${METHOD}/` and
  `results/ablations/${METHOD}/` accordingly. Each GPU script keeps **two separate `CONFIGS`
  arrays** (one per method) rather than a single generic list, since §6.1's config *basenames*
  differ between the two methods (rnhal: `rep_full`/`rep_spatial`/`rep_spectral`; texhal:
  `rep_q5`/`rep_q13`/`rep_q17`/`rep_full`) while §6.2/§6.3 are identical.
- **Reporting**: `dalmax/reporting/ablation_report.py` gained `STUDY_CONFIGS_BY_METHOD` (keyed
  `rnhal`/`texhal`) and a `--method` CLI flag (default `rnhal`); the pre-existing module-level
  `STUDY_CONFIGS` name is kept as an alias for `STUDY_CONFIGS_BY_METHOD["rnhal"]` for backward
  compatibility with existing callers/tests.
- **Makefile**: `METHOD ?= rnhal` threaded through `ablations-gpu{0,1}`, `ablations-all`,
  `ablations-colab`, `ablation-report`. A new `ablation-report-legacy` target is kept as its own
  entry point (`--root results/ablations --out docs/results/ablation_tables --method rnhal`, no
  `METHOD` substitution) so the already-executed RNHAL batch's historical reporting path is
  untouched by this reorganization.
- **Results/legacy compatibility**: the already-executed RNHAL batch's results
  (`results/ablations/{6_1,6_2,6_3}/`, no method segment) are left exactly where they are
  (append-only, `.claude/rules/data-safety.md`) — **not** moved under `results/ablations/rnhal/`.
  A fresh `METHOD=rnhal` invocation of the scripts writes to the *new* `results/ablations/rnhal/`
  root by default; `SKIP_EXISTING`'s glob is scoped to `RESULTS_ROOT` and therefore cannot see the
  legacy tree. Extending the legacy batch instead of starting a new one requires passing
  `RESULTS_ROOT=results/ablations` explicitly — documented in both scripts' headers and both
  runbooks, not silently handled, since silently redirecting a "new run" into an already-published
  results tree would be a reproducibility hazard (`.claude/rules/reproducibility.md`).
- **Docs**: the already-committed `docs/results/ablation_tables/*.{csv,md,tex}` (RNHAL's executed
  batch) stay at the top level unmoved; new per-method reports land under
  `docs/results/ablation_tables/{rnhal,texhal}/`.

## Consequences

**Positive:**
- TexHAL's ablation suite can be developed, smoke-tested, and (eventually) run on the lab machine
  without touching or risking RNHAL's already-executed, paper-cited results.
- The three-paper plan is now traceable from the folder structure itself, not just from spec prose.
- Every existing RNHAL command still works unchanged as long as `METHOD` is omitted (all defaults
  are `rnhal`), and the legacy results/report paths remain first-class (`ablation-report-legacy`),
  so this is additive, not a breaking change to the executed batch's provenance.

**Negative / follow-up:**
- Two GPU scripts now each carry two `CONFIGS` arrays instead of one — a third method (should one
  ever be added) would need a third array in each script rather than a fully generic
  config-discovery mechanism. Accepted as a reasonable trade-off for two methods; revisit if a
  third method is added.
- Operators must remember that `METHOD=rnhal` (the default) is a *new* results root, not the
  executed legacy one — mitigated by explicit warnings in both scripts' headers and both runbooks,
  but still a manual step, not automatically enforced.
- `dalmax/reporting/ablation_report.py`'s `STUDY_TITLES["6_3"]` was generalized from "Contribution
  of the two RNHAL stages" to "Contribution of the two stages" (method-agnostic) — a cosmetic
  change to any *newly generated* `ablation_6_3.md`/`.tex`; the already-committed
  `docs/results/ablation_tables/ablation_6_3.md`/`.tex` (RNHAL's executed batch) are unaffected
  unless someone re-runs `make ablation-report-legacy` and re-commits.
