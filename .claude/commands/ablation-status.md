---
description: Read the ablation study spec(s) and results/ablations/, and print a checklist of which ablation runs are implemented, executed, or pending, plus the aggregated F1 table where available.
---

**Method-aware since 2026-08-30**: there are two ablation suites, `rnhal` (paper 3, SSRAE,
`.specs/experiments/ablation-study.md`, **executed** 2026-08-26) and `texhal` (paper 2, VCTex,
`.specs/experiments/ablation-study-texhal.md`, materialized, not yet run) — see
`files_config/ablations/README.md` and `.specs/experiments/papers-roadmap.md`. Treat `$ARGUMENTS`
as `[method] [results_root]` (both optional): `method` defaults to `rnhal`, `results_root` defaults
to `results/ablations/<method>` (or `results/ablations` — the legacy root — when explicitly asked
to check the already-executed RNHAL batch). Read `.specs/experiments/ablation-study.md` (for
`rnhal`) or `.specs/experiments/ablation-study-texhal.md` (for `texhal`), mirrored by
`.claude/skills/ablation-study/SKILL.md`, and cross-reference against the results root's
"Materialized files" section to report ablation progress.

**Status (2026-08-30): rnhal has 11 configs, all executed 2026-08-26 (33/33 runs, zero failures) —
see `ablation-study.md`'s "Execution record". texhal has 12 configs (`files_config/ablations/
texhal/*.json` + `micro/*.json` CPU smoke mirrors, `make smoke-ablations` covers both suites), none
executed yet.** For `rnhal` this command's job is mostly confirmation (the batch is done); for
`texhal` its job is to report which of the 12 × 3 seeds have landed under `results/ablations/texhal/`
(none, as of this writing).

Steps:

1. Load the chosen method's spec's "Materialized files" section for its `(study, config)` pairs and
   their exact JSON/GPU-script mapping:
   - **rnhal, §6.1 Representation ablation** (`files_config/ablations/rnhal/rep_full.json`,
     `rep_spatial.json`, `rep_spectral.json`): SSRAE `Q=13`, reference hierarchy
     `n_clusters=[600,200,100]`.
   - **texhal, §6.1 Representation ablation** (`files_config/ablations/texhal/rep_q5.json`,
     `rep_q13.json`, `rep_q17.json`, `rep_full.json`): VCTex `Q` swept single-scale vs. multi-scale
     `[5,17]`, same reference hierarchy.
   - **§6.2 Hierarchy ablation** (`hier_L1.json` .. `hier_L4.json`, identical basenames for both
     methods): method's own full-scale embedding, `n_query=100`, hierarchy per the spec's run table.
   - **§6.3 Stage contribution** (`stage_full.json`, the shared "w/o representation" run `files_config/campaign/params_kmh.json` (ADR 0009, campaign only),
     `stage_no_hierarchy.json`, identical basenames for both methods): full method vs.
     ResNet-ImageNet + hierarchical vs. method's embedding + `flat_proportional`.
2. For each config, check two things:
   - **Implemented**: "yes" for every config in both suites as of 2026-08-30 — each has a
     `files_config/ablations/<method>/<config>.json` validated by `tests/test_ablation_configs.py`.
     Flag as regressed only if that test file or the JSON is missing/failing.
   - **Executed**: run `find results/ablations/<method> -path "*/<config>/*/SEED_*/NQ_*/RepresentationStrategy/results.json"`
     (or `find results/ablations -path "..."` for the rnhal legacy root) to count how many of the 3
     seeds have landed for that config. §6.3's `stage_full` row for `rnhal` may also be satisfied by
     the pre-Phase-2 reference runs (`results/dalmax{1,2}/.../SSRAEKmeansHCSampling/`, recomputed for
     macro F1 offline) — check both locations.
3. If any runs were found, run
   `poetry run python -m dalmax.reporting.ablation_report --root results/ablations/<method> --out <scratch dir> --method <method>`
   (or `--root results/ablations` with `--method rnhal` for the legacy root) and include the
   resulting `ablation_summary.csv` numbers (final-round macro F1 mean ± std per config) in the
   report — do not write into `paper_drafts/` or `docs/results/` from this read-only status
   command; use a scratch/tmp output directory instead, per `.claude/rules/data-safety.md`.
   For a quicker completeness check (per-triple OK/INCOMPLETE/MISSING against the full expected
   artifact set, not just a run count), use
   `poetry run python -m dalmax.reporting.results_doctor verify --root results/ablations --method
   <method>` instead/in addition — for `rnhal` it transparently also checks the legacy
   no-method-segment root, so it works whether or not that tree has been migrated (see
   `dalmax/reporting/results_doctor.py`, `COLAB_RUNBOOK.md` §9 / `LAB_RUNBOOK.md` §6).
4. Print a checklist:

```
## 6.1 Representation ablation
- [x] rep_full     — implemented: yes | executed: <n>/3 seeds | macro F1: <mean +/- std or "TBD">
- [ ] rep_spatial  — implemented: yes | executed: <n>/3 seeds | macro F1: <mean +/- std or "TBD">
- [ ] rep_spectral — implemented: yes | executed: <n>/3 seeds | macro F1: <mean +/- std or "TBD">

## 6.2 Hierarchy ablation
- [ ] hier_L1  (k=[50])              — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] hier_L2a (k=[300,100])         — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] hier_L2b (k=[100,50])          — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] hier_L3  (k=[300,100,50])      — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] hier_L4  (k=[300,100,50,25])   — implemented: yes | executed: <n>/3 seeds | macro F1: <status>

## 6.3 Stage contribution
- [ ] stage_full               — implemented: yes | executed: <n>/3 seeds (or reused dalmax{1,2}) | macro F1: <status>
- [ ] stage_no_representation  — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] stage_no_hierarchy       — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
```

5. End with a one-line summary of what to run next on the lab machine (e.g.
   `METHOD=<method> bash scripts/ablations/run_ablation_gpu_0.sh` / `run_ablation_gpu_1.sh`,
   checking `results/ablations/<method>/gpu{0,1}_failures.log` for any prior partial
   failures first), per `.claude/skills/running-experiments/SKILL.md`. If `<method>` is `texhal`
   and nothing has run yet, say so plainly — that suite's status is "not started," not partially
   complete.
