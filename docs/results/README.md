# Tracked experiment summaries

This folder is the **only** place experiment numbers come back into git. Raw runs stay
machine-local under `results/` (gitignored).

Workflow (any GPU machine, after a campaign batch -- see `.specs/experiments/campaign.md`):

```bash
make campaign-verify
make campaign-report               # writes docs/results/campaign/ (tables md/tex/csv, seed audit, confusion matrices)
git add docs/results/campaign/
git commit -m "exp: campaign results tables"
```

`docs/results/campaign/` holds, per paper, the md/tex tables (`paper1/`, `paper2/`, `paper3/`),
`summary.csv`, `seed_audit.md` and `confusion_matrices/` (small, text/PDF only). Never commit `.pkl`/`.pth`.

## Legacy (history)

`ablation_tables/` (`ablation_summary.csv` + `ablation_6_{1,2,3}.{md,tex}`) is the record of the executed
2026-08-26 RNHAL ablation batch (Colab, legacy layout `results/ablations/{6_1,6_2,6_3}/`). It is kept
as-is and never regenerated: the tool that produced it (`ablation_report.py`) was removed by ADR 0010,
and the campaign's `docs/results/campaign/paper3/` supersedes it once run. The paper skeleton that
consumed these tables lives in `paper_drafts/ablation_section.tex` (gitignored; see
`paper_drafts/README.md`).
