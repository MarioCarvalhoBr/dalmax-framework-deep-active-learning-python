# Use case: run the ablation study

Full specification: `experiments/ablation-study.md` (RNHAL, paper 3) and
`experiments/ablation-study-texhal.md` (TexHAL, paper 2) -- read the relevant one first, including its
"Materialized files" section mapping every run-table row to its params JSON.

**Since 2026-09-29 (ADR 0010) the ablations run only through the campaign** (`experiments/campaign.md`);
the per-GPU ablation scripts, `make ablations-*`/`ablation-report*`/`smoke-ablations` and
`dalmax.reporting.ablation_report` were removed. The configs are unchanged and still live in
`files_config/ablations/{rnhal,texhal}/` (+ `micro/` CPU mirrors, validated by
`tests/test_ablation_configs.py`); the manifest references them.

## Order of operations

1. **Pre-flight: smoke-test locally first.** `make campaign-smoke` runs every campaign group once on
   `DATA/daninhas_micro/` (CPU, seed 1; the two adversarial baselines are skipped by default), including
   all ablation configs through the same `RepresentationStrategy` code path as the full-scale runs.
   Confirm with `make campaign-verify MICRO=1 SEEDS=1`.
2. **Run on a GPU machine** (never locally -- no GPU here, `infrastructure/execution-environments.md`):
   ```bash
   make campaign-run PART=rnhal      # paper 3: 14 groups x 3 seeds = 42 runs
   make campaign-run PART=texhal     # paper 2: 10 groups x 3 seeds = 30 runs
   ```
   Skip-existing makes a relaunch resume (a job is done only when its leaf holds every artifact;
   `results.json` is written last). Failures go to `results/campaign/failures.log` and the batch continues.
3. **Verify**: `make campaign-verify PART=rnhal` (per-job OK/INCOMPLETE/MISSING + seed audit; exit 0/1/2).
4. **Report**: `make campaign-report` writes `docs/results/campaign/paper{2,3}/` tables (md/tex, weighted
   and macro F1, `TBD` for missing rows) plus `summary.csv` and mean confusion matrices. Papers 2/3
   comparison tables also need paper 1's `shared/*` runs (`make campaign-run PART=paper1`).

## Verification before trusting results

Check per run (`.claude/agents/experiment-auditor.md`): (a) `run_metadata.json` shows the intended
`embedding`/`selection` config, (b) the embedding cache file actually used
(`results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{hash}.pkl`) matches the
intended `Q`/variant/dataset, (c) the hierarchy matches the §6.2 row (`config.dataset.selection.hierarchy`),
(d) `all_f1_macro` (not the weighted `all_f1_score`) is what gets reported -- `research-rules/metrics.md`,
(e) `results/campaign/failures.log` is empty or every failure has been re-run, and (f)
`determinism.nondeterministic_op_warnings` in `run_metadata.json` is understood (GPU determinism is
best-effort, `research-rules/reproducibility.md`).

## Aggregation and paper hand-off

`dalmax.reporting.campaign_report` supersedes the old ablation reporter. Map the resulting
`docs/results/campaign/paper{2,3}/6_{1,2,3}.tex` tables to the paper per
`experiments/ablation-study.md`'s "Mapping to the paper's `\subsubsection`s" table. The 2026-08-26
RNHAL batch (33 runs, legacy `results/ablations/`) and its tables in `docs/results/ablation_tables/`
remain as history.
