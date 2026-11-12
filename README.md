# DalMax

**Deep Active Learning Laboratory for UAV Weed Recognition**

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![Poetry](https://img.shields.io/badge/dependency--management-poetry-60A5FA)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python/actions/workflows/ci.yml/badge.svg)](https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python/blob/main/notebooks/colab_runbook.ipynb)

DalMax is a PyTorch research framework for **pool-based deep active learning on UAV images of weeds in
sugarcane fields**, built for a PhD project at UFMS (Brazil). It supports the three papers of the thesis:

1. **A benchmark of deep active learning strategies**: 20 query strategies (the 12 of DeepAL plus the 8 that
   DeepAL+ adds) and **KMH** (hierarchical k-means over frozen ImageNet ResNet-50 features), at three query
   budgets, against a fully supervised upper bound.
2. **TexHAL**: VCTex color-texture descriptors + hierarchical k-means batch selection.
3. **RNHAL** (Randomized Network-guided Hierarchical Active Learning): SSRAE randomized-network
   spatio-spectral descriptors + hierarchical k-means batch selection, the main contribution.

![DalMax framework: data, round loop, pluggable components, run outputs and campaign engine](assets/active-learning-framework.png)

*Figure: the DalMax framework (round loop, registries, campaign engine). The figure predates the current
code: its pool/test sizes and run count are those of the first campaign on the original dataset, and its
strategy box shows only the DeepAL strategies. The text of this README is the current reference.*

## Contents

- [What DalMax is](#what-dalmax-is)
- [Status](#status)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Query strategies](#query-strategies)
- [Dataset](#dataset)
- [The campaign](#the-campaign)
- [Run outputs and inference tools](#run-outputs-and-inference-tools)
- [Repository map](#repository-map)
- [License](#license) / [How to cite](#how-to-cite) / [Contact](#contact) / [Acknowledgements](#acknowledgements)
- [References](#references)

## What DalMax is

In every round the loop trains a classifier (ResNet-50 [[22]](#ref-22) for the weed dataset) on the labeled
set, scores or embeds the unlabeled pool, selects a batch with a query strategy, adds the oracle's labels and
evaluates on a fixed test set (accuracy, weighted and macro precision/recall/F1, confusion matrix).

- **One entry point**: `tools/trainer.py`, a thin shim over `dalmax.cli.main()`.
- **One package**: all Python source lives in `dalmax/`. Datasets, models, query strategies, embedding
  providers and selection modules are plain dict registries; an unknown name fails with an explicit error.
- **Reproducibility features**:
  - *Seeding*: `dalmax/seeding.py` seeds `random`, NumPy and PyTorch once per run; every component that needs
    its own randomness derives a sub-seed from the run seed (`derive_seed`), and auxiliary networks of a strategy
    are initialised in an isolated generator so they do not shift the classifier's random stream.
  - *`run_metadata.json`* in every run: the fully resolved config, git commit, library versions, environment
    (including the GPU actually used) and a `determinism` block. Bit-identical results are guaranteed on CPU
    only; on GPU determinism is best-effort, so report seed-averaged numbers.
  - *Data-order contract* (`dalmax/data_order.py`): class folders and file names are sorted, so a pool index
    names the same image on every machine; `run_metadata.json` records hashes of both splits' order.
  - *Keyed embedding cache* (`dalmax/embeddings/cache.py`): keyed on dataset, extractor, `Q`, variant, split,
    data order and pool; the payload stores the image paths and a mismatch triggers recomputation.
  - *Campaign manifests* (`files_config/campaign/`): every run the papers need is declared once, executed with
    resume, verified (completeness, seed audit by initial image paths, data-order audit) and reported.

Specifications are the source of truth: [`.specs/00-overview.md`](.specs/00-overview.md) and
[`.specs/README.md`](.specs/README.md); architecture decisions are ADRs in [`.specs/adr/`](.specs/adr/README.md).

## Status

- A **first campaign** (main manifest and a KMH parameter sweep) was executed on Google Colab Pro between
  2026-09-30 and 2026-11-11 on the original dataset `DATA/daninhas_full`.
- Two defects were then found that affect those results:
  - **KI-46**: the original dataset contains byte-identical duplicate images, including images present in both
    train and test.
  - **KI-47**: files were listed in an unsorted, session-dependent order, so the same seed started from other
    images in different Colab sessions, and the embedding cache could serve features of other images.

  Both are documented in [`.specs/quality/known-issues.md`](.specs/quality/known-issues.md). The first
  campaign's numbers are superseded and are not reported here.
- A **clean re-run of everything** on the duplicate-free dataset `DATA/daninhas_full_fixed` is **prepared and
  being executed** ([ADR 0016](.specs/adr/0016-clean-rerun-fixed-dataset.md)). Its code version is pinned at the
  git tag `campaign-fixed-v2`, which the Colab notebook checks out. No results of the clean campaign are
  reported yet.

## Installation

DalMax is **Poetry-only** ([ADR 0001](.specs/adr/0001-adopt-poetry.md)).

1. **Prerequisites**: Python **3.10 to 3.12** (`requires-python = ">=3.10,<3.13"`) and `git`.
2. **Install Poetry**, preferably via [`pipx`](https://pipx.pypa.io/):

   ```bash
   sudo apt install pipx
   pipx ensurepath
   pipx install poetry
   ```

3. **Clone and install**:

   ```bash
   git clone https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python.git
   cd dalmax-framework-deep-active-learning-python
   poetry install        # or: make setup
   ```

`poetry install` creates an in-project virtualenv at `./.venv` from `poetry.lock` (exact pins, including
`torch==2.5.0` and `torchvision==0.20.0`). Prefix commands with `poetry run`. No GPU is needed for
development, tests or the CPU smoke run.

## Quick start

```bash
make lint     # ruff check
make format   # ruff format
make test     # pytest, fast tests only
make smoke    # builds DATA/daninhas_micro (needs DATA/daninhas_full), one CPU trainer.py run, then the fast tests
```

One experiment (the same configuration as a paper-1 entropy run; needs the dataset and, in practice, a GPU):

```bash
poetry run python tools/trainer.py \
    --dir_results results/scratch/ \
    --params_json files_config/campaign/params_paper1.json \
    --dataset_name DANINHAS \
    --strategy_name EntropySampling \
    --n_query 100 --n_init_labeled 100 --n_round 8 \
    --seed 1 --device cuda
```

Arguments (`poetry run python tools/trainer.py --help`): `--dir_results`, `--params_json`, `--seed`,
`--n_init_labeled`, `--n_query`, `--n_round`, `--dataset_name {CIFAR10,DANINHAS}`, `--strategy_name` (see the
next section), `--device {auto,cuda,cpu}` (pass `cuda` explicitly on GPU machines) and
`--embedding_variant {full,spatial,spectral}` (SSRAE only; overrides the params file).

The params JSON is keyed by dataset name and holds the training settings (`data_dir`, `n_epoch`, `n_drop`,
`n_classes`, loader and optimizer args) plus optional blocks: `embedding` / `selection` (for
`RepresentationStrategy`), `config_kmh` (legacy hierarchy, read as a hierarchical selection), `model_args`
(e.g. `dropout_p` for the MC-dropout strategies) and `strategy_args` (hyperparameters of the DeepAL+ ports).
The full schema is in [`.specs/experiments/experimental-protocol.md`](.specs/experiments/experimental-protocol.md);
the ablation configs are explained in [`files_config/ablations/README.md`](files_config/ablations/README.md).

## Query strategies

Every `--strategy_name` choice of `dalmax/query_strategies/registry.py`. "Changes training" says whether the
classifier itself is trained differently from the standard recipe (not only which images are queried).
Names marked **(+)** are the eight strategies ported from DeepAL+ [[16]](#ref-16)
([ADR 0015](.specs/adr/0015-deepalplus-strategies.md)); the twelve unmarked benchmark strategies come from
DeepAL [[26]](#ref-26).

| `--strategy_name` | Method | Family | Changes training | Ref. |
|---|---|---|---|---|
| `RandomSampling` | Random sampling | random baseline | no | |
| `LeastConfidence` | Least confidence | uncertainty | no | [[1]](#ref-1) |
| `MarginSampling` | Margin sampling | uncertainty | no | [[2]](#ref-2) |
| `EntropySampling` | Entropy sampling | uncertainty | no | [[3]](#ref-3) |
| `LeastConfidenceDropout` | Least confidence, MC dropout | uncertainty with MC dropout | yes: dropout layer | [[4]](#ref-4) [[5]](#ref-5) |
| `MarginSamplingDropout` | Margin, MC dropout | uncertainty with MC dropout | yes: dropout layer | [[4]](#ref-4) [[5]](#ref-5) |
| `EntropySamplingDropout` | Entropy, MC dropout | uncertainty with MC dropout | yes: dropout layer | [[4]](#ref-4) [[5]](#ref-5) |
| `BALDDropout` | BALD | uncertainty with MC dropout | yes: dropout layer | [[6]](#ref-6) [[5]](#ref-5) |
| `VarRatio` (+) | Variation ratio | uncertainty with MC dropout | yes: dropout layer | [[7]](#ref-7) [[5]](#ref-5) |
| `MeanSTD` (+) | Mean standard deviation | uncertainty with MC dropout | yes: dropout layer | [[8]](#ref-8) |
| `AdversarialBIM` | DFAL with the basic iterative method | adversarial margin | no | [[9]](#ref-9) [[10]](#ref-10) |
| `AdversarialDeepFool` | DFAL with DeepFool | adversarial margin | no | [[9]](#ref-9) [[11]](#ref-11) |
| `LossPredictionLoss` (+) | Learning loss | learned loss | yes: joint training with a loss-prediction module | [[12]](#ref-12) |
| `CEALSampling` (+) | CEAL | pseudo-labelling | yes: confident pool images pseudo-labelled for the next training | [[13]](#ref-13) |
| `KMeansSampling` | k-means sampling | diversity | no | [[14]](#ref-14) |
| `KCenterGreedy` | k-center greedy (core-set) | diversity | no | [[15]](#ref-15) |
| `KCenterGreedyPCA` (+) | k-center greedy on 50 PCA components | diversity | no | [[15]](#ref-15) [[16]](#ref-16) |
| `VAAL` (+) | Variational adversarial active learning | adversarial representation | no (trains its own VAE and discriminator) | [[17]](#ref-17) |
| `BadgeSampling` (+) | BADGE | hybrid | no | [[18]](#ref-18) [[19]](#ref-19) |
| `WAAL` (+) | Wasserstein adversarial active learning | hybrid | yes: min-max training with a critic | [[20]](#ref-20) |
| `WAALRandomQuery` | WAAL's training with a uniform random query | control | yes: as `WAAL` | [[20]](#ref-20) |
| `RepresentationStrategy` | Generic: embedding provider + selection module (KMH, TexHAL, RNHAL) | representation-based hierarchical selection | no | [[21]](#ref-21) |
| `SSRAEKmeansSampling` | Preset: SSRAE + flat k-means | representation-based selection (flat) | no | [[25]](#ref-25) |
| `VCTexKmeansSampling` | Preset: VCTex + flat k-means | representation-based selection (flat) | no | [[24]](#ref-24) |
| `SSRAEKmeansHCSampling` | Preset: SSRAE + hierarchical k-means | representation-based hierarchical selection | no | [[25]](#ref-25) [[21]](#ref-21) |
| `VCTexKmeansHCSampling` | Preset: VCTex + hierarchical k-means | representation-based hierarchical selection | no | [[24]](#ref-24) [[21]](#ref-21) |
| `FullSupervised` | Full supervision on the whole pool | upper bound | trains once on every pool label | |

Notes:

- **MC-dropout family**: the ResNet-50 has no dropout layer of its own, so these six strategies need
  `"model_args": {"dropout_p": ...}`; the campaign uses `files_config/campaign/params_paper1_mc_dropout.json`.
- **`WAALRandomQuery`** is not a benchmark strategy. It keeps WAAL's training, hyperparameters and seeded
  generators and replaces only the query by a uniform random draw. WAAL against the control isolates WAAL's
  selection at equal training; the control against `RandomSampling` isolates WAAL's training at equal (random)
  selection (ADR 0015, amendment of 2026-11-12).
- **`FullSupervised`** is not active learning: the whole pool is labeled at initialization, it requires
  `--n_round 0` and never queries. It is the upper bound of paper 1.
- **The lab's own methods** are parameterisations of `RepresentationStrategy`, set in the params JSON's
  `embedding` / `selection` blocks:
  - **KMH**: hierarchical k-means [[21]](#ref-21) over frozen ImageNet [[23]](#ref-23) ResNet-50
    [[22]](#ref-22) features (`resnet_imagenet`; `files_config/campaign/params_kmh.json`).
  - **TexHAL**: VCTex descriptors [[24]](#ref-24) (`vctex`; canonical `Q = 17`,
    [ADR 0011](.specs/adr/0011-texhal-canonical-q17.md); configs in `files_config/ablations/texhal_q17/`).
  - **RNHAL**: SSRAE randomized-network descriptors [[25]](#ref-25) (`ssrae`, `Q = 13`; configs in
    `files_config/ablations/rnhal/`).

  The four presets pin the legacy `Q` (SSRAE `13`, VCTex `(5, 17)`) and ignore `embedding.q`.
- **Deviations from DeepAL+** (detail and reasons in [ADR 0015](.specs/adr/0015-deepalplus-strategies.md)):
  - `VarRatio` implements the MC-dropout variation ratio (DeepAL+ ranks like least confidence).
  - `VAAL` uses the original method's discriminator targets (labeled 1, unlabeled 0), queries the lowest
    discriminator outputs and uses Adam for both networks.
  - `LossPredictionLoss` keeps the common `n_epoch` budget and detaches the module after a fraction of it.
  - `CEALSampling` uses the paper's threshold values.
  - All randomness is seeded from the run seed (PCA solver, k-means++ draws, auxiliary networks, tie-breaks).
  - `KMeansSamplingGPU` (faiss) was not ported.

## Dataset

The canonical dataset is `DATA/daninhas_full_fixed/`: UAV image patches of five weed classes of sugarcane
fields. It is built from the original `DATA/daninhas_full/` by
[`scripts/datasets/build_daninhas_full_fixed.py`](scripts/datasets/build_daninhas_full_fixed.py)
([ADR 0016](.specs/adr/0016-clean-rerun-fixed-dataset.md)), the only sanctioned way to create a dataset
directory. **The images are not distributed in this repository** (`DATA/` is gitignored and treated as
immutable input).

```
DATA/daninhas_full_fixed/
    train/<class>/<file>     training pool
    test/<class>/<file>      fixed test set, never queried
    README.txt               counts and the identity rule
    MANIFEST.csv             one row per image: split,class,file,md5
    logs/                    build provenance: build_history.txt, source_manifest.csv
```

| Folder | Class | Train | Test |
|---|---|---:|---:|
| `DATASET_BRACHIARIA` | *Brachiaria* spp. | 365 | 172 |
| `DATASET_COLONIAO` | *Panicum maximum* | 982 | 89 |
| `DATASET_GRAMINEA` | Gramineae spp. | 1,156 | 502 |
| `DATASET_MAMONA` | *Ricinus communis* | 3,056 | 167 |
| `DATASET_OUTRAS_FOLHAS_LARGAS` | Other broadleaf | 1,671 | 78 |
| **Total** | | **7,230** | **1,008** |

**Identity rule**: two files are the same image when their MD5 digests are equal. No MD5 occurs twice in the
dataset, neither inside a split nor across train and test. The test split keeps one file per distinct image of
the original test split. The train split keeps one file per distinct image of the original train split, minus
every image also in test and minus the near-duplicates of test images listed in
[`files_config/datasets/daninhas_full_fixed_near_duplicate_leaks.csv`](files_config/datasets/daninhas_full_fixed_near_duplicate_leaks.csv).

Verify a copy (read only): `make dataset-check`. It compares the files on disk with `MANIFEST.csv` and
checks for repeated MD5s. It must end with
`OK: files on disk match MANIFEST.csv, no repeated md5 inside or across splits`.

`DATA/daninhas_micro/` (a small stratified replica built by `make micro-dataset`) is used by the smoke tests.
CIFAR-10 is still accepted by the CLI (`--dataset_name CIFAR10`, layout `DATA/DATA_CIFAR10/{train,test}/{0..9}/`)
as a secondary benchmark, but the campaign does not use it.

## The campaign

The campaign is the only way the papers' runs are executed and reported. It is driven by the engine
`dalmax/campaign.py` from two committed manifests (generated by `scripts/campaign/build_manifest.py`). Both
run seeds 1 to 3 on `daninhas_full_fixed` into new results roots:

| Manifest | Content | Runs / groups | Results root |
|---|---|---|---|
| `files_config/campaign/manifest.json` | paper 1 (20 strategies + KMH at `n_query` 10/50/100), the WAAL control, the upper bound, the TexHAL/RNHAL ablations | 297 / 99 | `results/campaign_fixed/` |
| `files_config/campaign/manifest_kmh_sweep.json` | KMH parameter sweep ([ADR 0012](.specs/adr/0012-kmh-parameter-sweep.md)) | 165 / 55 | `results/campaign_kmh_fixed/` |

The 462 runs are split into two **GPU parts** of similar estimated cost, one per Colab account, run in parallel.
Counts are from `make campaign-list-gpu-part`; estimates from `files_config/campaign/gpu_time_estimates.json`:

| `GPU_PART` | Content | Runs / groups | Est. GPU h |
|---|---|---|---|
| `part_1_gpu` | paper 1 at nq100 (with the WAAL control), upper bound, all ablations, KMH sweep nq100 + nq50 | 285 / 95 | 49.9 |
| `part_2_gpu` | paper 1 at nq50 and nq10, KMH sweep nq10 | 177 / 59 | 48.8 |

Inside a part the plan runs cheap paper-1 groups first, then ablations, the KMH sweep, and the long baselines
(`AdversarialBIM`, `VAAL`/`WAAL`/`WAALRandomQuery`, `AdversarialDeepFool`) last.

```bash
make campaign-list-gpu-part   GPU_PART=part_1_gpu   # plan of both manifests, run counts, estimated GPU hours
make campaign-space-gpu-part  GPU_PART=part_1_gpu   # free space vs remaining runs (exit 4 when short; read only)
make campaign-pilot           GPU_PART=part_1_gpu   # first: seed 1 of the part's VAAL/WAAL + RandomSampling
make campaign-pilot-report    GPU_PART=part_1_gpu   # read only: wall time vs estimate, strategy diagnostics
make campaign-run-gpu-part    GPU_PART=part_1_gpu   # run the part; skips finished runs, re-run to resume
make campaign-verify-gpu-part GPU_PART=part_1_gpu   # OK/INCOMPLETE/MISSING + seed audit + data-order audit
make campaign-verify-all                            # after merging both parts: both manifests, every group
make campaign-report-all                            # tables -> docs/results/campaign_fixed/ (+ campaign_kmh_fixed/)
make campaign-smoke-gpu-part  GPU_PART=part_2_gpu   # micro plan of one part on CPU (seed 1)
```

Single-manifest targets (`campaign-list`, `campaign-run`, `campaign-verify`, `campaign-space`,
`campaign-report`) accept `PART=...`, `MANIFEST=path`, `MICRO=1`, `SEEDS=...`, `EXCLUDE=...`, `DEVICE=...` and
`RUN_ARGS=...`; see the comments in the [`Makefile`](Makefile) and
[`.specs/experiments/campaign.md`](.specs/experiments/campaign.md). Step-by-step operator guides:

- [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md) (section 6: the clean campaign on two accounts), or the runnable
  notebook [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb);
- [`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) for a lab machine.

Environment checks: `make lab-setup`, `make lab-check`, `make colab-setup`, `make colab-check`.
The code is hardware-agnostic: the GPU actually used is recorded only in each run's `run_metadata.json`
(`environment.gpus`).

## Run outputs and inference tools

Each run writes one leaf directory:

```
{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/
```

| File | Content |
|---|---|
| `results.json` | per-round metrics (accuracy, weighted and macro precision/recall/F1); written last, so its presence marks a finished run |
| `run_metadata.json` | resolved config, git commit, versions, environment, determinism settings, data-order hashes |
| `round_predictions.json` | test predictions of every round, with the test images' paths and true labels |
| `round_queries.json` | pool indices labeled in every round (round 0 = initial set), with labels and image paths |
| `predictions.csv` | last-round test predictions |
| `confusion_matrix.pdf`, `accuracy.pdf`, `precision.pdf`, `recall.pdf`, `f1_score.pdf` | plots |
| `log-dalmax.log` | run log |
| `saved_model.pth` | trained model in the `dalmax-checkpoint` format (`dalmax/models/checkpoint.py`) |

`results/` is append-only. Checkpoints written before 2026-08-23 hold no weights and cannot be loaded (KI-22).

Inference tools ([ADR 0006](.specs/adr/0006-checkpoint-format-and-inference-tools.md),
[`tools/README.md`](tools/README.md)) use the same preprocessing as training:

```bash
poetry run python tools/loader.py  --model <leaf>/saved_model.pth                     # inspect a checkpoint
poetry run python tools/predict.py --model <leaf>/saved_model.pth --image path/to/image.jpg
poetry run python tools/predict.py --model <leaf>/saved_model.pth \
    --dir DATA/daninhas_full_fixed/test/DATASET_GRAMINEA --out results/predictions/   # folder, recursive
poetry run python tools/gui.py                                                       # tkinter mini-app
```

`predict.py` writes `predictions.csv` (one probability column per class) and a `<stem>.pred.json` per image
into `--out` (default: next to the input); `--device {auto,cpu,cuda}` as in `trainer.py`.

## Repository map

| Path | Content |
|---|---|
| `tools/` | CLI entry points: `trainer.py`, `predict.py`, `loader.py`, `gui.py` |
| `dalmax/` | the package: `cli.py`, `campaign.py`, `seeding.py`, `data_order.py`, `config/`, `data/`, `models/`, `query_strategies/`, `embeddings/`, `selection/`, `experiment/`, `reporting/`, `inference/`, `tools/` (vendored SSRAE, VCTex and hierarchical k-means code) |
| `files_config/` | params files and manifests: `campaign/`, `ablations/`, `datasets/`, `benchmark/` (legacy, used by tests), `params_micro.json` |
| `scripts/` | dataset builder, micro dataset, manifest builders, Colab setup |
| `notebooks/` | `colab_runbook.ipynb` (runnable Colab guide) |
| `tests/` | pytest suite (registries, seeding, cache, campaign, golden runs, hardware-agnostic check) |
| `assets/` | README figure (TikZ source and PNG) and the qualification document |
| `.specs/` | specifications (source of truth), experiment protocols, known issues, ADRs in [`.specs/adr/`](.specs/adr/README.md) |
| `.claude/`, `CLAUDE.md`, `AGENTS.md` | agent configuration and operational guide for coding sessions |
| `DATA/`, `results/`, `docs/` | datasets, run outputs and generated reports (not tracked, except committed ablation tables under `docs/results/`) |

Module-level detail: [`.specs/architecture/current-state.md`](.specs/architecture/current-state.md).
CI (`.github/workflows/ci.yml`) runs the fast checks.

## License

DalMax is released under the MIT License. See [LICENSE](LICENSE). The vendored hierarchical k-means code in
`dalmax/tools/SSL/` carries its own Meta Platforms copyright header.

## How to cite

If you use this code, please cite the repository:

```bibtex
@misc{Carvalho2024dalmax,
  author       = {Mário de Araújo Carvalho},
  title        = {DalMax: Framework for Deep Active Learning Approaches},
  howpublished = {\url{https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python}},
  month        = {dec},
  year         = {2024},
  note         = {Available on GitHub}
}
```

The PhD qualification document with the theoretical foundation of DalMax/RNHAL is in
[`assets/phd_files/qualificacao-doutorado-mario-carvalho.pdf`](assets/phd_files/qualificacao-doutorado-mario-carvalho.pdf):

```bibtex
@phdthesis{Carvalho2025qualificacao,
  author     = {Mário de Araújo Carvalho and Wesley Nunes Gonçalves},
  title      = {Deep Active Learning for Precision Agriculture: A Computational Approach},
  school     = {Universidade Federal de Mato Grosso do Sul, Faculdade de Computação},
  year       = {2025},
  month      = {mar},
  type       = {Qualificação de Doutorado},
  address    = {Campo Grande, MS, Brazil},
  note       = {PhD Qualification Examination}
}
```

## Contact

- Author: Mário de Araújo Carvalho (mariodearaujocarvalho@gmail.com)
- Advisor: Wesley Nunes Gonçalves (UFMS)
- Project: <https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python>

## Acknowledgements

- The twelve classical strategies and the original loop are based on
  [DeepAL: Deep Active Learning in Python](https://github.com/ej0cl6/deep-active-learning) [[26]](#ref-26).
- The eight strategies marked (+) are ports of [DeepAL+](https://github.com/SineZHAN/deepALplus) (MIT licence)
  [[16]](#ref-16), with the deviations listed above.
- The hierarchical k-means and hierarchical sampling code is vendored from
  [facebookresearch/ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation) [[21]](#ref-21).

## References

1. <a id="ref-1"></a>David D. Lewis, William A. Gale. A Sequential Algorithm for Training Text Classifiers. *SIGIR '94: Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, 1994. <https://doi.org/10.1007/978-1-4471-2099-5_1>
2. <a id="ref-2"></a>Tobias Scheffer, Christian Decomain, Stefan Wrobel. Active Hidden Markov Models for Information Extraction. *Advances in Intelligent Data Analysis (IDA 2001)*, LNCS 2189, 2001. <https://doi.org/10.1007/3-540-44816-0_31>
3. <a id="ref-3"></a>Burr Settles. Active Learning Literature Survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009. <https://burrsettles.com/pub/settles.activelearning.pdf>
4. <a id="ref-4"></a>Yarin Gal, Zoubin Ghahramani. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48, 2016. <https://proceedings.mlr.press/v48/gal16.html>
5. <a id="ref-5"></a>Yarin Gal, Riashat Islam, Zoubin Ghahramani. Deep Bayesian Active Learning with Image Data. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 2017. <https://proceedings.mlr.press/v70/gal17a.html>
6. <a id="ref-6"></a>Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, Máté Lengyel. Bayesian Active Learning for Classification and Preference Learning. arXiv:1112.5745, 2011. <https://doi.org/10.48550/arXiv.1112.5745>
7. <a id="ref-7"></a>Linton C. Freeman. *Elementary Applied Statistics: For Students in Behavioral Science*. Wiley, New York, 1965. <https://openlibrary.org/works/OL5743680W>
8. <a id="ref-8"></a>Michael Kampffmeyer, Arnt-Borre Salberg, Robert Jenssen. Semantic Segmentation of Small Objects and Modeling of Uncertainty in Urban Remote Sensing Images Using Deep Convolutional Neural Networks. *2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2016. <https://doi.org/10.1109/cvprw.2016.90>
9. <a id="ref-9"></a>Melanie Ducoffe, Frederic Precioso. Adversarial Active Learning for Deep Networks: a Margin Based Approach. arXiv:1802.09841, 2018. <https://doi.org/10.48550/arXiv.1802.09841>
10. <a id="ref-10"></a>Alexey Kurakin, Ian Goodfellow, Samy Bengio. Adversarial examples in the physical world. arXiv:1607.02533, 2016. <https://doi.org/10.48550/arXiv.1607.02533>
11. <a id="ref-11"></a>Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard. DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. <https://doi.org/10.1109/CVPR.2016.282>
12. <a id="ref-12"></a>Donggeun Yoo, In So Kweon. Learning Loss for Active Learning. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019. <https://doi.org/10.1109/cvpr.2019.00018>
13. <a id="ref-13"></a>Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, Liang Lin. Cost-Effective Active Learning for Deep Image Classification. *IEEE Transactions on Circuits and Systems for Video Technology* 27(12), 2017. <https://doi.org/10.1109/tcsvt.2016.2589879>
14. <a id="ref-14"></a>Fedor Zhdanov. Diverse mini-batch Active Learning. arXiv:1901.05954, 2019. <https://doi.org/10.48550/arXiv.1901.05954>
15. <a id="ref-15"></a>Ozan Sener, Silvio Savarese. Active Learning for Convolutional Neural Networks: A Core-Set Approach. *International Conference on Learning Representations (ICLR)*, 2018. <https://openreview.net/forum?id=H1aIuk-RW>
16. <a id="ref-16"></a>Xueying Zhan, Qingzhong Wang, Kuan-hao Huang, Haoyi Xiong, Dejing Dou, Antoni B. Chan. A Comparative Survey of Deep Active Learning. arXiv:2203.13450, 2022. <https://doi.org/10.48550/arXiv.2203.13450> (code: <https://github.com/SineZHAN/deepALplus>)
17. <a id="ref-17"></a>Samrath Sinha, Sayna Ebrahimi, Trevor Darrell. Variational Adversarial Active Learning. *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019. <https://doi.org/10.1109/iccv.2019.00607>
18. <a id="ref-18"></a>Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal. Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds. *International Conference on Learning Representations (ICLR)*, 2020. <https://openreview.net/forum?id=ryghZJBKPS>
19. <a id="ref-19"></a>David Arthur, Sergei Vassilvitskii. k-means++: The Advantages of Careful Seeding. *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 2007. <https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf>
20. <a id="ref-20"></a>Changjian Shui, Fan Zhou, Christian Gagné, Boyu Wang. Deep Active Learning: Unified and Principled Method for Query and Training. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 108, 2020. <https://proceedings.mlr.press/v108/shui20a.html>
21. <a id="ref-21"></a>Huy V. Vo, Vasil Khalidov, Timothée Darcet, Théo Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo Touvron, Camille Couprie, Maxime Oquab, Armand Joulin, Hervé Jégou, Patrick Labatut, Piotr Bojanowski. Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach. arXiv:2405.15613, 2024. <https://doi.org/10.48550/arXiv.2405.15613>
22. <a id="ref-22"></a>Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. <https://doi.org/10.1109/CVPR.2016.90>
23. <a id="ref-23"></a>Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei. ImageNet: A large-scale hierarchical image database. *2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009. <https://doi.org/10.1109/CVPR.2009.5206848>
24. <a id="ref-24"></a>Ricardo T. Fares, Lucas C. Ribas. Volumetric Color-Texture Representation for Colorectal Polyp Classification in Histopathology Images. *Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)*, 2025. <https://doi.org/10.5220/0013315800003912>
25. <a id="ref-25"></a>Ricardo T. Fares, Lucas C. Ribas. A New Approach to Learn Spatio-Spectral Texture Representation with Randomized Networks: Application to Brazilian Plant Species Identification. *Engineering Applications of Neural Networks (EANN 2024)*, CCIS, 2024. <https://doi.org/10.1007/978-3-031-62495-7_33>
26. <a id="ref-26"></a>Kuan-Hao Huang. DeepAL: Deep Active Learning in Python. arXiv:2111.15258, 2021. <https://arxiv.org/abs/2111.15258> (code: <https://github.com/ej0cl6/deep-active-learning>)
