---
description: Read the ablation study spec and results/ablations/, and print a checklist of which ablation runs are implemented, executed, or pending, plus the aggregated F1 table where available.
---

Read `.specs/experiments/ablation-study.md` (mirrored by
`.claude/skills/ablation-study/SKILL.md`) and cross-reference against
`results/ablations/` (Phase 3's materialized layout — see that spec's
"Materialized files" section) to report ablation progress.

**Status (Phase 3 landed, 2026-08-23): all 11 configs are implemented as
`files_config/ablations/*.json` (+ `files_config/ablations/micro/*.json` CPU
smoke mirrors); `make smoke-ablations` passes 11/11 locally.** What remains
is running `scripts/ablations/run_ablation_gpu_0.sh` /
`run_ablation_gpu_1.sh` on the lab machine — this command's job is to report
which of the 11 × 3 seeds have actually landed under `results/ablations/`.

Steps:

1. Load `.specs/experiments/ablation-study.md`'s "Materialized files" section
   for the 11 `(study, config)` pairs and their exact JSON/GPU-script
   mapping:
   - **6.1 Representation ablation** (`files_config/ablations/rep_full.json`,
     `rep_spatial.json`, `rep_spectral.json`): SSRAE `Q=13`, reference
     hierarchy `n_clusters=[600,200,100]`.
   - **6.2 Hierarchy ablation** (`hier_L1.json` .. `hier_L4.json`): SSRAE
     full, `n_query=100`, hierarchy per `ablation-study.md`'s run table.
   - **6.3 RNHAL stage contribution** (`stage_full.json`,
     `stage_no_representation.json`, `stage_no_hierarchy.json`): full RNHAL
     vs. ResNet-ImageNet + hierarchical vs. SSRAE + `flat_proportional`.
2. For each of the 11 configs, check two things:
   - **Implemented**: always "yes" as of Phase 3 — every config has a
     `files_config/ablations/<config>.json` validated by
     `tests/test_ablation_configs.py`. Flag as regressed only if that test
     file or the JSON is missing/failing.
   - **Executed**: run `find results/ablations -path "*/<config>/*/SEED_*/NQ_*/RepresentationStrategy/results.json"`
     (scoped to `$ARGUMENTS` as the results root if given, default
     `results/ablations`) to count how many of the 3 seeds have landed for
     that config. §6.3's `stage_full` row may also be satisfied by the
     pre-Phase-2 reference runs (`results/dalmax{1,2}/.../SSRAEKmeansHCSampling/`,
     recomputed for macro F1 offline) — check both locations.
3. If any runs were found, run
   `poetry run python -m dalmax.reporting.ablation_report --root results/ablations --out <scratch dir>`
   and include the resulting `ablation_summary.csv` numbers (final-round
   macro F1 mean ± std per config) in the report — do not write into
   `paper_drafts/` from this read-only status command; use a scratch/tmp
   output directory instead, per `.claude/rules/data-safety.md`.
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

## 6.3 RNHAL stage contribution
- [ ] stage_full               — implemented: yes | executed: <n>/3 seeds (or reused dalmax{1,2}) | macro F1: <status>
- [ ] stage_no_representation  — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
- [ ] stage_no_hierarchy       — implemented: yes | executed: <n>/3 seeds | macro F1: <status>
```

5. End with a one-line summary of what to run next on the lab machine (e.g.
   `bash scripts/ablations/run_ablation_gpu_0.sh` / `run_ablation_gpu_1.sh`,
   checking `results/ablations/gpu{0,1}_failures.log` for any prior partial
   failures first), per `.claude/skills/running-experiments/SKILL.md`.
