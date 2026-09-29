---
description: Report campaign progress (files_config/campaign/manifest.json vs results/campaign/): per-part run counts, OK/INCOMPLETE/MISSING, seed audit. Read-only.
---

Report the state of the campaign (ADR 0008/0009/0010, `.specs/experiments/campaign.md`). Treat `$ARGUMENTS`
as `[part]` (optional; `all` | `paper1` | `upper_bound` | `rnhal` | `texhal`, comma-separated allowed;
default `all`). This command is read-only: it never runs training and never writes under `results/`.

Steps:

1. `poetry run python -m dalmax.campaign list --part <part>` -- the expected job table and the counts
   (full scale: 192 runs / 64 groups: `paper1` 117, `upper_bound` 3, `rnhal` 42, `texhal` 30).
2. `poetry run python -m dalmax.campaign verify --part <part>` -- per job OK / INCOMPLETE / MISSING
   (`dalmax/reporting/leaf_check.py` artifact check) plus the seed-consistency audit. Exit code:
   0 all OK, 1 an incomplete/missing job or a FAIL audit, 2 only audit WARNs.
3. Summarize per part: `<ok>/<expected>` runs, list every INCOMPLETE/MISSING job label, any audit
   FAIL/WARN offender, and the last lines of `results/campaign/failures.log` if it exists.
4. If runs exist, mention that `make campaign-report` builds the tables (it writes under
   `docs/results/campaign/`, so do NOT run it from this read-only command; use `--out` with a
   scratch directory if numbers are needed: `poetry run python -m dalmax.reporting.campaign_report
   --root results/campaign --out <scratch dir>`).
5. End with one line: what to run next (e.g. `make campaign-run PART=<missing part>` on the GPU machine;
   resume is safe, only incomplete jobs execute). Never name a GPU model: the GPU used is whatever
   `environment.gpus[0].name` says in the run's `run_metadata.json`.

The executed 2026-08-26 RNHAL ablation batch (33 runs) lives at the legacy root `results/ablations/`
with its committed tables in `docs/results/ablation_tables/`; it is history, not part of this status.
