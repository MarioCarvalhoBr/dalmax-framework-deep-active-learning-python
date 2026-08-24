# Baseline results — where they live

`results/` is **gitignored and local-only** (0 files under `results/` are
tracked in git as of this writing). This file documents, read-only, what was
observed on disk in this environment on 2026-08-23; it is not a substitute
for the lab machine's actual state and must be re-verified there before the
ablation runs (§ `ablation-study.md`) are executed, since results can differ
between the local checkout and the lab/Colab machines that produced them.

## Top-level `results/` listing (observed)

```
results/cifar10/
results/cifar10_ssra_6126_vctex_6107/
results/dalmax1/
results/dalmax2/
results/dalmax2_vctex_8053/
results/dalmax2_vctex_8134/
results/dalmax2_vctex_8386/
results/features_dict_ssrae.pkl        (132 MB — SSRAE embedding cache, no cache key)
results/features_dict_vctex.pkl        (209 MB — VCTex embedding cache, no cache key)
results/logs/
results/olds_results/
results/original_indices.txt           (2.5 MB — id;class_name;path dump from Data.create_indexes_path)
results/Y_train.pkl                    (labels cache)
results/zzz_bkp_cifar10/
```

Only `dalmax1/` and `dalmax2/` were inspected in depth per this batch's
scope; the others are named suggestively (`*_vctex_<run-id>`,
`olds_results`, `zzz_bkp_*`) but their exact contents are **TBD** —
flag for a future `experiment-auditor` pass before relying on them.

## `results/dalmax1/`

Path: `results/dalmax1/daninhas_full/`. Produced by `scripts/benchmark/run_pipe_gpu_0.sh`
(`files_config/benchmark/params_df_gpu_0.json`, GPU 0, `config_kmh: n_clusters=[600,200,100],
n_levels=3, sample_sizes=[30,15,2]`).

Structure:
```
results/dalmax1/daninhas_full/
├── SEED_{1,2,3}/
│   └── NQ_{10,50,100}_NIL_100_NR_8_NE_10/
│       └── {strategy}/            # results.json, predictions.csv, plots, log, model
├── data_results.json              # aggregated per-seed/per-NQ/per-method summary (all_acc/precision/recall/f1_score)
└── results/                       # output of the reporting pipeline (utils/report/ at the time
                                    # these results were generated; moved to dalmax/reporting/ in Phase 4)
    ├── AVERAGES/{NQ_10,NQ_50,NQ_100}_.../   # seed-averaged tables/plots
    └── SEED_{1,2,3}/
```

Strategies present under each `SEED_x/NQ_y_.../` (verified for
`SEED_1/NQ_10_.../` and `SEED_1/NQ_50_.../`): `RandomSampling`,
`LeastConfidence`, `MarginSampling`, `EntropySampling`,
`LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`,
`KMeansSampling`, `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`,
`AdversarialDeepFool`, `SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling` —
i.e. the **full baseline set plus both RNHAL-family hierarchical
strategies**, across all three `n_query` values and all three seeds. This is
broader than what `scripts/benchmark/run_pipe_gpu_0.sh` alone would produce (that script only
runs `SSRAEKmeansHCSampling`), so `dalmax1/` reflects more than one script
invocation over time — **TBD** reconcile the exact provenance/commit of each
strategy's runs (`log-dalmax.log` per leaf directory should carry a
timestamp; not inspected in this batch).

`data_results.json` sample entry (`SEED_1 → NQ_100_NIL_100_NR_8_NE_10 →
AdversarialBIM`): `all_acc=0.8295`, `all_precision=0.8935`,
`all_recall=0.8295`, `all_f1_score=0.8442` — these look like **single scalar
final-round values** (not per-round lists as in the raw `results.json`),
i.e. this file is a post-processed summary, likely produced by
`utils/report/2_report_build_chunk_results.py` (now `dalmax/reporting/chunk_results.py`,
renamed/moved in Phase 4) or a similar script — TBD
confirm exactly which script writes `data_results.json` (not one of the
`dalmax/reporting/*.py` scripts inspected directly in this batch).

**This is the RNHAL reference source for ablation §6.3** ("RNHAL (full)"
row) — use the `SSRAEKmeansHCSampling` entries at `n_query=100` across the
three seeds.

## `results/dalmax2/`

Path: `results/dalmax2/daninhas_full/`. Produced by `scripts/benchmark/run_pipe_gpu_1.sh`
(`files_config/benchmark/params_df_gpu_1.json`, GPU 1, `config_kmh: n_clusters=[500,200,150],
n_levels=3, sample_sizes=[60,30,2]`).

Structure mirrors `dalmax1/` (`SEED_{1,2,3}/NQ_{10,50,100}_NIL_100_NR_8_NE_10/`)
but **only contains `SSRAEKmeansHCSampling`** — consistent with
`scripts/benchmark/run_pipe_gpu_1.sh` running exactly that one strategy across the
`n_query × seed` grid, nothing else. No baseline strategies, no VCTex
variant, no `results/` (report-pipeline output) subfolder or
`data_results.json` observed under `dalmax2/` in this pass — TBD confirm
whether the report pipeline was ever run against `dalmax2/` (its scripts
default `--input_dir` to `results/dalmax1/daninhas_full/...`, see
`experimental-protocol.md`).

`dalmax1/` and `dalmax2/` together represent the **two GPUs' worth of the
same nominal SSRAEKmeansHCSampling protocol under two different
`config_kmh` hierarchy configurations** — useful as an informal starting
point for the hierarchy ablation (§6.2) but not a substitute for it: neither
matches any of the `L=1..4` rows in the ablation table (both use `n_levels=3`
with different `n_clusters`/`sample_sizes`).

## Pointer to paper tables

The paper under revision, `phd_files/Artigo_melhorias_Active_Learning_Mario.pdf`,
presumably reports the main RNHAL-vs-baseline comparison from `dalmax1/`
(it has the full baseline set). **TBD**: which exact table/run selection in
the PDF corresponds to which `results/` directory and seed-averaging
methodology — not verified against the PDF text in this batch; a
`paper-liaison`-style cross-check (owned by another batch) should confirm
this before the ablation numbers are added alongside it.
