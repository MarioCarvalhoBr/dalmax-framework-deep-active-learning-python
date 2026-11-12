---
description: Report campaign progress (files_config/campaign/manifest.json vs results/campaign_fixed/, plus the KMH sweep vs results/campaign_kmh_fixed/): per-part / per-GPU-part run counts, OK/INCOMPLETE/MISSING, seed audit. Read-only.
---

Report the state of the campaign (ADR 0008/0009/0010, `.specs/experiments/campaign.md`). Treat `$ARGUMENTS`
as `[part]` (optional; `all` | `paper1` | `upper_bound` | `rnhal` | `texhal` | `texhal_q17` | `hier_transfer`,
comma-separated allowed, or a GPU part `part_1_gpu` | `part_2_gpu`; default `all`). This command is read-only: it never
runs training and never writes under `results/`. Since ADR 0016 the manifests describe the clean re-run on
`DATA/daninhas_full_fixed`; pass both manifests: `M=files_config/campaign/manifest.json,files_config/campaign/manifest_kmh_sweep.json`.

Steps:

1. `poetry run python -m dalmax.campaign list --manifest $M --part <part>` -- the expected job table, the counts and
   the estimated GPU hours (full scale: main 294 runs / 98 groups: `paper1` 189, `upper_bound` 3, `rnhal` 42,
   `texhal` 30, `texhal_q17` 18, `hier_transfer` 12; KMH sweep 165 / 55; GPU parts `part_1_gpu` 282 runs,
   `part_2_gpu` 177 runs).
2. `poetry run python -m dalmax.campaign verify --manifest $M --part <part>` -- per job OK / INCOMPLETE / MISSING
   (`dalmax/reporting/leaf_check.py` artifact check) plus the seed-consistency audit (round-0 image paths)
   and the data-order audit (KI-47); with several manifests also the intentional duplicate-pair report.
   Exit code: 0 all OK, 1 an incomplete/missing job or a FAIL audit, 2 only audit WARNs.
3. Summarize per part: `<ok>/<expected>` runs, list every INCOMPLETE/MISSING job label, any audit
   FAIL/WARN offender, and the last lines of `results/campaign_fixed/failures.log` and
   `results/campaign_kmh_fixed/failures.log` if they exist.
4. If runs exist, mention that `make campaign-report` builds the tables (it writes under
   `docs/results/campaign_fixed/`, so do NOT run it from this read-only command; use `--out` with a
   scratch directory if numbers are needed: `poetry run python -m dalmax.reporting.campaign_report
   --root results/campaign_fixed --out <scratch dir>`).
5. End with one line: what to run next (e.g. `make campaign-run-gpu-part GPU_PART=<part>` on the GPU machine;
   resume is safe, only incomplete jobs execute). Never name a GPU model: the GPU used is whatever
   `environment.gpus[0].name` says in the run's `run_metadata.json`.

The executed 2026-08-26 RNHAL ablation batch (33 runs) lives at the legacy root `results/ablations/`
with its committed tables in `docs/results/ablation_tables/`; it is history, not part of this status.
