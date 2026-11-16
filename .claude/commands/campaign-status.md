---
description: Report campaign progress (files_config/campaign/manifest.json vs results/campaign_fixed/, plus the KMH sweep vs results/campaign_kmh_fixed/): per-part / per-GPU-part run counts, OK/INCOMPLETE/MISSING, seed audit. Read-only.
---

Report the state of the campaign (ADR 0008/0009/0010, `.specs/experiments/campaign.md`). Treat `$ARGUMENTS`
as `[part]` (optional; `all` | `paper1` | `paper1_control` | `upper_bound` | `rnhal` | `texhal` | `texhal_q17` | `hier_transfer` | `texhal_q13` | `descriptors_nq` | `seeds_ext` | `kmh_nq100` | `kmh_nq50` | `kmh_nq10` | `kmh_wide` | `kmh_norm`,
comma-separated allowed, or a GPU part `part_1_gpu` | `part_2_gpu`; default `all`). This command is read-only: it never
runs training and never writes under `results/`. Since ADR 0016 the manifests describe the clean re-run on
`DATA/daninhas_full_fixed`; pass both manifests: `M=files_config/campaign/manifest.json,files_config/campaign/manifest_kmh_sweep.json`.

Steps:

1. `poetry run python -m dalmax.campaign list --manifest $M --part <part>` -- the expected job table, the counts and
   the estimated GPU hours (full scale: main 357 runs / 119 groups: `paper1` 189, `paper1_control` 3 (the WAAL
   control `WAALRandomQuery`), `upper_bound` 3, `rnhal` 42, `texhal` 30, `texhal_q17` 18, `hier_transfer` 12,
   `texhal_q13` 24 (ADR 0017), `descriptors_nq` 12 and `seeds_ext` 24 (ADR 0020; `seeds_ext` runs seeds 4-6);
   KMH sweep 231 / 77: `kmh_nq100` 63, `kmh_nq50` 51, `kmh_nq10` 51, `kmh_wide` 39 (ADR 0018), `kmh_norm` 27
   (ADR 0019); GPU parts `part_1_gpu` 336 runs, `part_2_gpu` 252 runs; 588 in total).
2. `poetry run python -m dalmax.campaign verify --manifest $M --part <part>` -- per job OK / INCOMPLETE / MISSING
   (`dalmax/reporting/leaf_check.py` artifact and integrity check: a truncated/empty file shows as
   `INCOMPLETE ... (integrity: ...)`), the code-version audit (distinct `git_commit` of the OK leaves; WARN on more
   than one or a missing one, no effect on the exit code), the seed-consistency audit (round-0 image paths) and the
   data-order audit (KI-47; also FAIL for a plan leaf under a foreign dataset folder); with several manifests also
   the intentional duplicate-pair report. Exit code: 0 all OK, 1 an incomplete/missing job or a FAIL audit, 2 only
   seed/data-order audit WARNs.
3. Summarize per part: `<ok>/<expected>` runs, list every INCOMPLETE/MISSING job label (with its integrity reason),
   the distinct commits (expected and explained: `464a5cc` = v2, the v3 commit for `texhal_q13`, the v4 commit for
   `kmh_wide`, the v5 commit for `kmh_norm`, the v6 commit for `descriptors_nq` / `seeds_ext`; ADR
   0017/0018/0019/0020; anything else is unexplained), any audit FAIL/WARN (the seed audit also lists seeds 4-6,
   whose runs are the `seeds_ext` groups only)
   offender, and the last lines of `results/campaign_fixed/failures.log` and
   `results/campaign_kmh_fixed/failures.log` if they exist. For `kmh_norm` leaves also check that the round-0 weighted
   F1 equals the executed KMH runs of the same seed (0.3839 / 0.4783 / 0.2469 for seeds 1 / 2 / 3, KI-50 kept by
   ADR 0019).
4. If runs exist, mention that `make campaign-report` builds the tables (it writes under
   `docs/results/campaign_fixed/`, so do NOT run it from this read-only command; use `--out` with a
   scratch directory if numbers are needed: `poetry run python -m dalmax.reporting.campaign_report
   --root results/campaign_fixed --out <scratch dir>`).
5. End with one line: what to run next (e.g. `make campaign-run-gpu-part GPU_PART=<part>` on the GPU machine;
   resume is safe, only incomplete jobs execute). Never name a GPU model: the GPU used is whatever
   `environment.gpus[0].name` says in the run's `run_metadata.json`.

The executed 2026-08-26 RNHAL ablation batch (33 runs) lives at the legacy root `results/ablations/`
with its committed tables in `docs/results/ablation_tables/`; it is history, not part of this status.
