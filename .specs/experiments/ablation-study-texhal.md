# Ablation study specification — TexHAL (paper 2)

> **Tooling note (2026-09-29, ADR 0010).** References below to `scripts/ablations/*`, `scripts/benchmark/*`,
> `make ablations-*`/`ablation-report*`/`smoke-ablations`, `dalmax/reporting/ablation_report.py`,
> `results_doctor` or a `METHOD` variable describe tooling that was **removed**; these configs now run
> and report only through the campaign (`make campaign-run PART=rnhal|texhal`, `make campaign-report`,
> `make campaign-smoke`; `experiments/campaign.md`). The configs, protocol, run tables and the
> execution records are unchanged. CLI entry points live in `tools/` (`tools/trainer.py`).

Status: **materialized and CPU-smoke-tested (2026-08-30, 12/12 passing, ~3m24s local CPU) — not
yet run on real hardware.** This is the TexHAL analog of
[`ablation-study.md`](ablation-study.md) (the RNHAL/paper-3 spec) — see
[`papers-roadmap.md`](papers-roadmap.md) for how the two papers/methods relate. Every run table
below is `TBD` pending a real lab-machine or Colab run; the run tables' *rows* and exact params
JSON/CLI syntax are final.

TexHAL = **Tex**ture-guided **H**ierarchical **A**ctive **L**earning: the VCTex color-texture
representation (see "VCTex background" below) + hierarchical k-means batch selection — the same
selection module RNHAL uses (`dalmax/selection/hierarchical_kmeans.py`), swapped onto a different
embedding provider. CLI preset: `VCTexKmeansHCSampling` (unchanged); every ablation cell below uses
the generic `RepresentationStrategy` instead (`--strategy_name RepresentationStrategy`), since the
preset ignores the params JSON's `"embedding"`/`"selection"` blocks entirely (same caveat as
RNHAL's §6.1, see `ablation-study.md`).

All three sub-studies are evaluated on the held-out `test/` split of `daninhas_full`, using seeds
and budgets consistent with `experimental-protocol.md`, and report **both weighted and macro F1**
(same "which F1 is primary" framing as `ablation-study.md` — weighted primary / macro secondary,
pending final advisor sign-off, applies identically here once numbers exist).

## VCTex background (verified 2026-08-26/30)

**Provenance**: per the VCTex method authors (Ricardo T. Fares and Lucas C. Ribas, São Paulo State
University / UNESP), email to Mário Carvalho, 2026-08-26, and the original manuscript
`phd_files/artigo-original-tecnica-vctex-manuscript.pdf` — "Volumetric Color-Texture
Representation for Colorectal Polyp Classification in Histopathology Images."

- VCTex slides **volumetric 3×3×3 color cubes** over the RGB image (`dalmax/tools/VCTex/extractor.py`:
  `split()` extracts 3×3 windows per channel, then `torch.vstack`s the three channels into a
  `(27, H*W)` input matrix `X` — 27 = 3×3×3, confirmed in code) and flattens them as inputs to a
  Randomized Autoencoder (RAE, `dalmax/tools/VCTex/rnn.py::RNN`).
- The representation is the **flattened decoder (output-layer) weights**, `beta.flatten()` — the
  same "representation-as-learned-weights" idea as SSRAE, but over a different (volumetric,
  27-dim-per-patch) input rather than SSRAE's per-channel 9-dim-per-patch blocks.
- **Single parameter `Q`** (the RAE's hidden-layer/latent-space size) controls the representation's
  richness — VCTex has no separate spatial/spectral split the way SSRAE does (SSRAE's
  `slice_embedding` full/spatial/spectral logic is SSRAE-only, see
  `dalmax/query_strategies/representation.py`'s guard on `embedding_provider.name == "ssrae"`).
- **Embedding dimension = `27 * (Q + 1)` per RAE** (27 input dims × `(Q+1)` output columns per
  the `(p, Q+1)` beta shape pattern shared with SSRAE's `(9, Q+1)` blocks, verified against
  `dalmax/tools/VCTex/extractor.py`'s `X.shape[0] == 27` and `RNN(Q=..., p=X.shape[0], ...)`), summed
  across every `Q` value when multiple scales are concatenated (`dalmax/embeddings/vctex_provider.py`
  passes `Q=q_list` and stacks each scale's output). Concretely:

  | `Q` | Embedding dim (`27*(Q+1)`) |
  |---|---|
  | `Q=5` | 162 |
  | `Q=13` | 378 |
  | `Q=17` | 486 |
  | `Q=[5,17]` (concatenated) | 648 |

  The `648` figure for `Q=[5,17]` matches the value stated by the method authors — this is the
  paper's best-parameters multi-scale setting (`dalmax/query_strategies/registry.py::
  REPRESENTATION_PRESET_Q["vctex"] = (5, 17)`, the existing `VCTexKmeansSampling`/
  `VCTexKmeansHCSampling` preset default, unchanged by this ablation work). Per the authors, `Q=13`
  and `Q=17` alone are also good smaller-vector single-scale settings — motivating this ablation's
  4-row §6.1 (see below), one more row than RNHAL's 3-row spatial/spectral/full split.
- **Verification status**: the `27*(Q+1)` formula was derived from reading the vendored
  `dalmax/tools/VCTex/extractor.py`/`rnn.py` code, and **empirically confirmed exactly** (2026-08-30,
  `make smoke-ablations`) against `RepresentationStrategy`'s logged embedding shape for every `q`
  value this suite uses — see "VCTex-through-generic-path findings" below for the measured table.

## 6.1 Representation ablation

Unlike RNHAL's §6.1 (which slices one fixed-`Q` SSRAE embedding into spatial/spectral column-groups
via `embedding.variant`), TexHAL's representation module has no such split — VCTex's `Q` is itself
the tunable hyperparameter, and it is a *scale*, not a slicing axis. This ablation therefore sweeps
`Q` directly: two single-scale settings (`Q=5`, the existing `VCTexKmeansSampling`-adjacent small
scale, and `Q=13`/`Q=17`, the method authors' other cited good single-scale settings) against the
multi-scale combination (`Q=[5,17]`, the paper's best-parameters setting, labeled "Full" for
structural parity with RNHAL's table).

`selection` is held fixed at the **same reference hierarchy** as RNHAL's §6.1
(`n_clusters=[600,200,100]`, `n_levels=3`, `sample_sizes=[30,15,2]`, matching
`params_df_gpu_0.json`'s `config_kmh`) for cross-paper/cross-method comparability at the same
hierarchy shape.

**Confirmed (2026-09-01)**: TexHAL's ablation reference hierarchy is confirmed to match RNHAL's
reference hierarchy (`n_clusters=[600,200,100]`, `n_levels=3`) for cross-paper comparability (confirmed
by user 2026-09-01). The §6.1/§6.2/§6.3 reference-hierarchy choice is final.

### 6.1 run table

| Variant | embedding.q | n_query | seeds | strategy_name | n_round | n_epoch | F1 (weighted, mean±std) | F1 (macro, mean±std) |
|---|---|---|---|---|---|---|---|---|
| Q=5           | `[5]`     | 100 | 1,2,3 | RepresentationStrategy | 8 | 10 | TBD | TBD |
| Q=13          | `[13]`    | 100 | 1,2,3 | RepresentationStrategy | 8 | 10 | TBD | TBD |
| Q=17          | `[17]`    | 100 | 1,2,3 | RepresentationStrategy | 8 | 10 | TBD | TBD |
| Full (Q=[5,17]) | `[5, 17]` | 100 | 1,2,3 | RepresentationStrategy | 8 | 10 | TBD | TBD |

### 6.1 exact params JSON + CLI

One params JSON per variant, differing only in `embedding.q` (`Q=5` shown; swap the `q` array for
`[13]`/`[17]`/`[5, 17]`):

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10, "n_drop": 10, "n_classes": 5,
        "train_args": {"batch_size": 256, "num_workers": 4},
        "test_args": {"batch_size": 256, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "vctex", "q": [5], "variant": "full"},
        "selection": {
            "method": "hierarchical",
            "hierarchy": {"n_clusters": [600, 200, 100], "n_levels": 3, "sample_sizes": [30, 15, 2]}
        }
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> poetry run python tools/trainer.py \
    --params_json files_config/ablations/texhal/rep_q5.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablations/texhal/6_1/rep_q5/ --device cuda
```

`variant` is always `"full"` — `slice_embedding` (SSRAE-only) is never applied to a VCTex
embedding; see `files_config/ablations/README.md`.

## 6.2 Hierarchy ablation

Identical structure and run-table shape to RNHAL's §6.2 (`ablation-study.md`) — same
`n_clusters`/`n_levels`/`sample_sizes` grid, `n_query=100` for every row — with `embedding` fixed
at VCTex multi-scale (`q: [5, 17]`, `variant: "full"`) instead of SSRAE full.

### 6.2 run table

| Variant | selection.hierarchy.n_levels | selection.hierarchy.n_clusters | selection.hierarchy.sample_sizes | n_query | seeds | strategy_name | F1 (weighted, mean±std) | F1 (macro, mean±std) |
|---|---|---|---|---|---|---|---|---|
| L=1     | 1 | [50]               | [3]           | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |
| L=2 (a) | 2 | [300, 100]         | [15, 5]       | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |
| L=2 (b) | 2 | [100, 50]          | [5, 3]        | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |
| L=3     | 3 | [300, 100, 50]     | [15, 5, 3]    | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |
| L=4     | 4 | [300, 100, 50, 25] | [15, 5, 3, 2] | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |
| Reference (§6.1's "Full" row) | 3 | [600, 200, 100] | [30, 15, 2] | 100 | 1,2,3 | RepresentationStrategy | TBD | TBD |

### 6.2 exact params JSON + CLI (L=3 shown)

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10, "n_drop": 10, "n_classes": 5,
        "train_args": {"batch_size": 256, "num_workers": 4},
        "test_args": {"batch_size": 256, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "vctex", "q": [5, 17], "variant": "full"},
        "selection": {
            "method": "hierarchical",
            "hierarchy": {"n_clusters": [300, 100, 50], "n_levels": 3, "sample_sizes": [15, 5, 3]}
        }
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> poetry run python tools/trainer.py \
    --params_json files_config/ablations/texhal/hier_L3.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablations/texhal/6_2/hier_L3/ --device cuda
```

## 6.3 Contribution of the two TexHAL stages

- **TexHAL (full)**: VCTex `q=[5,17]` + hierarchical selection, reference hierarchy —
  `stage_full.json`.
- **Without representation module**: keep hierarchical selection, replace VCTex embeddings with
  **ImageNet-pretrained ResNet embeddings** (`dalmax/embeddings/resnet_imagenet_provider.py`) —
  `files_config/campaign/params_kmh.json` (the former `stage_no_representation.json`; neither
  method's representation module is involved in this row, so **it is one run shared by papers 1
  (KMH@100), 2 and 3** -- ADR 0009, 2026-09-29; the campaign's `shared/kmh_nq100`).
- **Without hierarchical module**: VCTex embeddings + `flat_proportional` (flat k-means,
  proportional-random per cluster) — `stage_no_hierarchy.json`.

### 6.3 run table

| Variant | Representation | Selection | strategy_name / config | F1 (weighted, mean±std) | F1 (macro, mean±std) |
|---|---|---|---|---|---|
| TexHAL (full)             | VCTex, Q=[5,17]           | Hierarchical k-means | `RepresentationStrategy`, `stage_full.json` | TBD | TBD |
| w/o representation module | ImageNet ResNet50 penult. | Hierarchical k-means | `RepresentationStrategy`, `files_config/campaign/params_kmh.json` (shared run) | TBD | TBD |
| w/o hierarchical module   | VCTex, Q=[5,17]           | Flat k-means, proportional-random | `RepresentationStrategy`, `stage_no_hierarchy.json` | TBD | TBD |

**Superseded by ADR 0009 (2026-09-29)** -- ~~Confirmed (2026-09-01): `stage_no_representation` is NOT
shared between the RNHAL and TexHAL papers; each paper runs its own independently, with separate copies
(`files_config/ablations/{rnhal,texhal}/stage_no_representation.json`).~~ The 2026-09-01 decision is
reversed: the config is byte-identical, running it twice wastes GPU time and yields two numbers for
one computation. It is now ONE run (`shared/kmh_nq100` in the campaign manifest) consumed by paper 1's
KMH row and by the 6.3 tables of papers 2 and 3; the redundant copies were removed.

### 6.3 exact params JSON + CLI

**Row 3 — without hierarchical module** (VCTex `q=[5,17]` + flat proportional-random):

```json
{
    "DANINHAS": {
        "data_dir": "DATA/daninhas_full/",
        "n_epoch": 10, "n_drop": 10, "n_classes": 5,
        "train_args": {"batch_size": 256, "num_workers": 4},
        "test_args": {"batch_size": 256, "num_workers": 4},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        "embedding": {"extractor": "vctex", "q": [5, 17], "variant": "full"},
        "selection": {"method": "flat_proportional"}
    }
}
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> poetry run python tools/trainer.py \
    --params_json files_config/ablations/texhal/stage_no_hierarchy.json --dataset_name DANINHAS \
    --strategy_name RepresentationStrategy --n_query 100 --seed <1|2|3> \
    --n_round 8 --dir_results results/ablations/texhal/6_3/stage_no_hierarchy/ --device cuda
```

## VCTex-through-generic-path findings

`make smoke-ablations` (or `METHOD=texhal make smoke-ablations`) is the **first real execution of
VCTex through the generic `RepresentationStrategy` path** — previously VCTex only ran through the
legacy `VCTexKmeansSampling`/`VCTexKmeansHCSampling` presets (`dalmax/query_strategies/registry.py`).
`VCTexProvider` (`dalmax/embeddings/vctex_provider.py`) was already documented as CPU-capable
(unlike SSRAE, `dalmax.tools.VCTex.rnn.RNN` never hardcodes `.cuda()`), so no CUDA-only blocker was
expected going in.

**Result (2026-08-30, local CPU dev notebook, `METHOD=texhal make smoke-ablations`): 12/12 passed,
no `VCTexProvider` bug found.** No code change to `dalmax/embeddings/vctex_provider.py` was needed
— `VCTexProvider` worked correctly through the generic `RepresentationStrategy` path on the first
try, for every `q` value used by this suite. Wall time: **~3m24s** for all 12 configs (comparable
to `METHOD=rnhal`'s ~3m22s for its 11 configs, run back-to-back in the same session).

The `27*(Q+1)` formula was **empirically confirmed exactly** from `RepresentationStrategy`'s logged
embedding shape (`dalmax/query_strategies/representation.py`'s "Embedding matrix shape after
slicing" log line) for every `q` value in this suite, over the real ~796-image
`DATA/daninhas_micro/` unlabeled pool:

| `q` (as loaded) | Logged shape | Expected (`27*(Q+1)`, summed per scale) |
|---|---|---|
| `(5,)`     | `(796, 162)` | `27*(5+1) = 162` |
| `(13,)`    | `(796, 378)` | `27*(13+1) = 378` |
| `(17,)`    | `(796, 486)` | `27*(17+1) = 486` |
| `(5, 17)`  | `(796, 648)` | `27*6 + 27*18 = 162 + 486 = 648` |

The `648` figure matches both the method authors' stated value and this empirical measurement
exactly — the derivation in "VCTex background" above is now verified, not just code-derived.

## Output artifacts

Same artifact set and results-directory convention as RNHAL's ablation batch (`ablation-study.md`'s
"Output artifacts" section), rooted one level deeper by method:
`results/ablations/texhal/{study}/{config}/{dataset_folder}/SEED_{seed}/
NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/RepresentationStrategy/`. Aggregation:
`poetry run python -m dalmax.reporting.ablation_report --root results/ablations/texhal --out
docs/results/ablation_tables/texhal --method texhal` (`make ablation-report METHOD=texhal`).

## Materialized files (2026-08-30)

| §  | Row | Params JSON | GPU script (METHOD=texhal) |
|----|-----|--------------|------------|
| 6.1 | Q=5 | `files_config/ablations/texhal/rep_q5.json` | `run_ablation_gpu_0.sh` |
| 6.1 | Q=13 | `files_config/ablations/texhal/rep_q13.json` | `run_ablation_gpu_1.sh` |
| 6.1 | Q=17 | `files_config/ablations/texhal/rep_q17.json` | `run_ablation_gpu_1.sh` |
| 6.1 | Full (Q=[5,17]) | `files_config/ablations/texhal/rep_full.json` | `run_ablation_gpu_0.sh` |
| 6.2 | L=1, k=[50] | `files_config/ablations/texhal/hier_L1.json` | `run_ablation_gpu_0.sh` |
| 6.2 | L=2, k=[300,100] | `files_config/ablations/texhal/hier_L2a.json` | `run_ablation_gpu_1.sh` |
| 6.2 | L=2, k=[100,50] | `files_config/ablations/texhal/hier_L2b.json` | `run_ablation_gpu_0.sh` |
| 6.2 | L=3, k=[300,100,50] | `files_config/ablations/texhal/hier_L3.json` | `run_ablation_gpu_1.sh` |
| 6.2 | L=4, k=[300,100,50,25] | `files_config/ablations/texhal/hier_L4.json` | `run_ablation_gpu_1.sh` |
| 6.3 | TexHAL (full) | `files_config/ablations/texhal/stage_full.json` | `run_ablation_gpu_1.sh` |
| 6.3 | w/o representation module | `files_config/campaign/params_kmh.json` (shared run, ADR 0009) | campaign only |
| 6.3 | w/o hierarchical module | `files_config/ablations/texhal/stage_no_hierarchy.json` | `run_ablation_gpu_1.sh` |

11 configs / 33 runs by the ablation-only scripts (`SEEDS=(1 2 3)`), GPU 0 = 4 configs, GPU 1 = 7 configs (the 12th row, "w/o representation module", is the campaign's shared run, ADR 0009) — see
`scripts/ablations/run_ablation_gpu_{0,1}.sh`'s headers for the split rationale.
`files_config/ablations/texhal/micro/` mirrors all 12 for `make smoke-ablations`, validated by
`tests/test_ablation_configs.py` (parametrized over both `rnhal/` and `texhal/`, 46 files total).

## Mapping to the paper's `\subsubsection`s

| This spec section | Paper subsubsection (working title) |
|---|---|
| §6.1 Representation ablation | "Representation ablation" / effect of VCTex's multi-scale `Q` |
| §6.2 Hierarchy ablation | "Hierarchy ablation" / effect of hierarchy depth `L` and cluster counts `k_i` |
| §6.3 Contribution of the two TexHAL stages | "Contribution of the two TexHAL stages" / ablating the representation and hierarchical-selection modules independently |

TBD: confirm exact subsubsection titles once the advisor finalizes paper 2's outline.
