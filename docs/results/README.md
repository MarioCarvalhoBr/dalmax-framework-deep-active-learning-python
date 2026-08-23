# Tracked experiment summaries

This folder is the **only** place experiment numbers come back into git. Raw runs stay
machine-local under `results/` (gitignored).

Workflow (lab machine, after an ablation batch — see `scripts/ablations/`):

```bash
python -m dalmax.reporting.ablation_report --root results/ablations --out docs/results/ablation_tables
git add docs/results/ablation_tables/
git commit -m "Add Phase 3 ablation study results"
git push
```

Contents committed here: `ablation_summary.csv` + `ablation_6_{1,2,3}.{md,tex}` (small, text-only).
The paper skeleton that consumes these tables lives in `paper_drafts/ablation_section.tex`
(gitignored; see `paper_drafts/README.md`).
