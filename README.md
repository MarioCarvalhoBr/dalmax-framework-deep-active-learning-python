# DalMax

**Deep Active Learning Laboratory for UAV Weed Recognition**

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![Poetry](https://img.shields.io/badge/dependency--management-poetry-60A5FA)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/CI-pending-lightgrey)

![Deep Active Learning Framework](assets/active-learning-framework.png)

## Overview

DalMax is the research codebase for a PhD project (UFMS, Brazil) on **Deep Active
Learning applied to UAV-based weed recognition in precision agriculture**. It
implements and compares acquisition (query) strategies for training a deep
classifier (ResNet50) under a limited annotation budget, on both a domain dataset
(`daninhas_full`, UAV weed image patches) and CIFAR-10 as a secondary benchmark.

The main scientific contribution is **RNHAL (Randomized Network-guided
Hierarchical Active Learning)**. RNHAL combines two stages:

1. A **randomized-network spatio-spectral representation module**, instantiated
   through the **SSRAE** formulation (see
   [`phd_files/artigo-original-tecnica-ssrae-manuscript.pdf`](phd_files/artigo-original-tecnica-ssrae-manuscript.pdf)
   and `core/tools/SSRAE/`). For an image `x`, closed-form randomized autoencoders
   produce a spatial signature per channel (`Θ_R`, `Θ_G`, `Θ_B`) and a spectral
   signature per adjacent channel pair (`Ω_RG`, `Ω_GB`, `Ω_BR`); the final embedding
   is conceptually their concatenation `Φ(x) = [Θ_R, Θ_G, Θ_B, Ω_RG, Ω_GB, Ω_BR]`,
   encoding intra-channel spatial structure (`Θ_R, Θ_G, Θ_B`) and inter-channel
   spectral dependency (`Ω_RG, Ω_GB, Ω_BR`). **Implementation note (verified
   2026-08-23)**: the actual vector produced by `core/tools/SSRAE/extractor.py` is
   row-interleaved, not laid out as two contiguous halves — naive slicing such as
   `emb[:len(emb)//2]` does **not** recover the spatial-only signatures. See
   `.specs/experiments/ablation-study.md` §6.1 for the corrected slicing used by the
   representation ablation.
2. A **hierarchical k-means batch selection mechanism** (`core/tools/SSL/`) that
   recursively partitions the embedding space of the unlabeled pool and samples the
   query batch across the induced hierarchy, aiming for batches that are both
   informative and structurally diverse rather than redundant in visually dense
   regions.

The full formal definition is in
[`phd_files/Active_Learning_Mario/method_full.tex`](phd_files/Active_Learning_Mario/method_full.tex);
the paper under revision is
[`phd_files/Artigo_melhorias_Active_Learning_Mario.pdf`](phd_files/Artigo_melhorias_Active_Learning_Mario.pdf).

For the project's specifications, architecture, and experiment protocols, see
[`.specs/00-overview.md`](.specs/00-overview.md) and [`.specs/README.md`](.specs/README.md).

## Implemented query strategies

These are the exact `--strategy_name` choices exposed by `demo.py`.

**Uncertainty-based**
- **Random Sampling** — select samples randomly (baseline).
- **Least Confidence** — select samples where the model is least confident [1].
- **Margin Sampling** — select samples where the margin between the two most
  likely classes is smallest [2].
- **Entropy Sampling** — select samples where the prediction entropy is highest [3].
- **Least Confidence / Margin / Entropy Sampling (Dropout)** — MC-dropout variants
  of the above, estimating uncertainty via stochastic forward passes [4].
- **BALD Dropout** — Bayesian Active Learning by Disagreement, using MC-dropout to
  estimate epistemic uncertainty [4].

**Diversity / representation-based**
- **KMeans Sampling** — select samples representative of feature-space clusters [5].
- **KCenter Greedy** — core-set selection minimizing the covering radius over the
  labeled + unlabeled pool [5].

**Adversarial**
- **Adversarial BIM** — select samples closest to the decision boundary via the
  Basic Iterative Method [6].
- **Adversarial DeepFool** — select samples with the smallest adversarial margin,
  estimated via DeepFool [6].

**Proposed RNHAL family (SSRAE / VCTex representation + k-means selection)**
- **SSRAE KMeans Sampling** — flat k-means over SSRAE spatio-spectral embeddings.
- **VCTex KMeans Sampling** — flat k-means over VCTex texture embeddings.
- **SSRAE KMeans HC Sampling** — hierarchical k-means (RNHAL) over SSRAE embeddings.
- **VCTex KMeans HC Sampling** — hierarchical k-means (RNHAL) over VCTex embeddings.

See [`.specs/experiments/experimental-protocol.md`](.specs/experiments/experimental-protocol.md)
for the exact protocol (seeds, budgets, metrics) and
[`.specs/experiments/ablation-study.md`](.specs/experiments/ablation-study.md) for the
ablation design that isolates each RNHAL stage.

## Installation

Python **3.10–3.12** is required (upper-bounded by the `torch==2.5.0` pin). The
project historically documented Python 3.9; that constraint is stale and superseded
by the Poetry configuration below.

### Poetry (primary)

```bash
poetry install
```

`poetry.toml` pins the virtualenv to `./.venv` (`in-project = true`). Dependency
versions are exact-pinned in `pyproject.toml` for reproducibility (research lab,
not a library). See [`make setup`](#development) for the wrapped version of this
command.

### pip (fallback, e.g. lab machine / Colab without Poetry)

```bash
pip install -r requirements.txt
```

`requirements.txt` is a **generated export** (`make export-reqs`) from
`pyproject.toml` — do not hand-edit it; regenerate it instead.

### CUDA

Training on `daninhas_full`/CIFAR10 with `torch==2.5.0` / `torchvision==0.20.0`
targets **CUDA 12.4** on the lab GPUs. No GPU is required for local development,
smoke tests, or CPU-only strategies.

## Datasets

### daninhas_full (primary)

UAV weed image patches, 5 classes, `train/`/`test/` split:

```
DATA/
    daninhas_full/
        train/
            DATASET_BRACHIARIA/
            DATASET_COLONIAO/
            DATASET_GRAMINEA/
            DATASET_MAMONA/
            DATASET_OUTRAS_FOLHAS_LARGAS/
        test/
            DATASET_BRACHIARIA/
            DATASET_COLONIAO/
            DATASET_GRAMINEA/
            DATASET_MAMONA/
            DATASET_OUTRAS_FOLHAS_LARGAS/
```

`DATA/` is treated as immutable, read-only input (see
[`.claude/rules/data-safety.md`](.claude/rules/data-safety.md)); it is not
distributed in this repository. See
[`.specs/research-rules/dataset-protocol.md`](.specs/research-rules/dataset-protocol.md)
for class definitions and any known imbalance notes.

### CIFAR-10 (secondary benchmark)

```bash
mkdir -p DATA
# Download DATA_CIFAR10.zip:
# https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing
unzip DATA_CIFAR10.zip -d DATA
```

Expected layout:

```
DATA/
    DATA_CIFAR10/
        train/
            0/  1/  2/  ...  9/
        test/
            0/  1/  2/  ...  9/
```

## Usage

Entry point: `demo.py`. Example (RNHAL / SSRAE hierarchical strategy on the weed dataset):

```bash
python demo.py \
    --dir_results results/dalmax1/ \
    --params_json params_df_gpu_0.json \
    --dataset_name DANINHAS \
    --strategy_name SSRAEKmeansHCSampling \
    --n_query 100 \
    --n_init_labeled 100 \
    --n_round 8 \
    --seed 1
```

CLI arguments: `--dir_results`, `--params_json`, `--seed`, `--n_init_labeled`,
`--n_query`, `--n_round`, `--dataset_name {CIFAR10,DANINHAS}`, `--strategy_name`
(one of the strategies listed above).

### Params JSON schema

Hyperparameters are keyed by dataset name (see `params_df_gpu_0.json` /
`params_df_gpu_1.json`, one file per lab GPU):

```json
{
  "DANINHAS": {
    "data_dir": "DATA/daninhas_full/",
    "n_epoch": 10,
    "n_drop": 10,
    "n_classes": 5,
    "train_args": { "batch_size": 256, "num_workers": 4 },
    "test_args": { "batch_size": 256, "num_workers": 4 },
    "optimizer_args": { "lr": 0.05, "momentum": 0.3 },
    "config_kmh": {
      "n_clusters": [600, 200, 100],
      "n_levels": 3,
      "sample_sizes": [30, 15, 2]
    }
  },
  "CIFAR10": {
    "data_dir": "DATA/DATA_CIFAR10/",
    "n_epoch": 20,
    "n_drop": 10,
    "n_classes": 10,
    "train_args": { "batch_size": 64, "num_workers": 1 },
    "test_args": { "batch_size": 1000, "num_workers": 1 },
    "optimizer_args": { "lr": 0.05, "momentum": 0.3 }
  }
}
```

`config_kmh` (hierarchical k-means config, only needed by the `*HCSampling`
strategies) sets the number of clusters per hierarchy level (`n_clusters`), the
number of levels (`n_levels`), and how many samples are drawn per level
(`sample_sizes`).

### Results directory convention

`demo.py` writes to:

```
{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/
```

containing `results.json`, `predictions.csv`, `confusion_matrix.pdf`,
`accuracy.pdf`/`precision.pdf`/`recall.pdf`/`f1_score.pdf`, `log-dalmax.log`, and
the saved model checkpoint. See
[`.specs/experiments/experimental-protocol.md`](.specs/experiments/experimental-protocol.md).

## Execution environments

DalMax is developed and run across three environments:

1. **Local dev notebook** — 16 GB RAM, no GPU, Python 3.12, Poetry. Used for coding,
   CPU smoke tests on tiny subsets, and report generation.
2. **Lab machine (primary training)** — 2× NVIDIA GPUs, 10 GB each. Workflow: push
   from the notebook, pull on the lab machine, run `run_pipe_gpu_0.sh` /
   `run_pipe_gpu_1.sh` (one params JSON per GPU), results come back via git or copy.
3. **Google Colab Pro (secondary/burst)** — one-off runs; the dataset is uploaded as
   a zip and extracted into the runtime rather than read file-by-file from Drive.

Details, decision matrix, and the Colab checklist:
[`.specs/infrastructure/execution-environments.md`](.specs/infrastructure/execution-environments.md).

## Development

This project uses a Claude Code multi-agent setup for coding sessions — see
[`CLAUDE.md`](CLAUDE.md) for the operational guide, [`.claude/`](.claude/) for
agents/commands/rules/skills, and [`.specs/`](.specs/) for the source-of-truth
specifications.

```bash
make setup         # poetry install
make lint          # ruff check
make format        # ruff format
make test          # pytest (fast tests only)
make smoke         # proxy smoke check (fast tests); true micro-run is a Phase 1 deliverable
make export-reqs   # regenerate requirements.txt from pyproject.toml
```

## Repository layout

```
dalmax-deep-active-learning-python/
├── demo.py                    # CLI entry point (training + AL loop)
├── core/                      # models, query strategies, RNHAL tools (SSRAE, SSL)
│   ├── query_strategies/      # one module per acquisition strategy
│   └── tools/
│       ├── SSRAE/             # randomized-network spatio-spectral extractor
│       └── SSL/               # hierarchical k-means selection
├── utils/                     # dataset handlers, orchestrator (registries), report/
├── params_df_gpu_0.json       # hyperparameters for lab GPU 0
├── params_df_gpu_1.json       # hyperparameters for lab GPU 1
├── run_pipe_gpu_0.sh          # experiment batch runner, GPU 0
├── run_pipe_gpu_1.sh          # experiment batch runner, GPU 1
├── scripts/                   # baseline pipelines, report utilities
├── DATA/                      # datasets (gitignored, immutable)
├── results/                   # experiment outputs (gitignored)
├── phd_files/                 # PhD documents, paper LaTeX sources, references
├── .claude/                   # multi-agent config: agents, commands, rules, skills
├── .specs/                    # specifications: architecture, experiments, ADRs
├── tests/                     # pytest suite (imports, registry, SSRAE layout)
├── CLAUDE.md                  # operational guide for Claude Code sessions
├── AGENTS.md                  # tool-agnostic mirror of CLAUDE.md
└── Makefile
```

## Roadmap

The codebase is being refactored in phases to make the ablation study
(representation ablation, hierarchy ablation, RNHAL-stage-contribution ablation)
a matter of configuration rather than new code:

**Phase 1 — Safety net → Phase 2 — Core refactor → Phase 3 — Ablations → Phase 4 — Polish.**

Full phase plan and acceptance criteria:
[`.specs/architecture/refactor-plan.md`](.specs/architecture/refactor-plan.md).
Ablation design: [`.specs/experiments/ablation-study.md`](.specs/experiments/ablation-study.md).

## License

DalMax is released under the MIT License. See [LICENSE](LICENSE).

## Citing

If you use this code in your research or applications, please consider citing the
repository:

```bibtex
@misc{Carvalho2024dalmax,
  author       = {Mário de Araújo Carvalho},
  title        = {DalMax: Framework for Deep Active Learning Approaches},
  howpublished = {\url{https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python}},
  month        = {dec},
  year         = {2024},
  note         = {Available on GitHub},
  annote       = {A Python framework for implementing and comparing deep active learning methods.}
}
```

This repository also contains the PhD qualification document presenting the
theoretical foundation and methodology behind DalMax/RNHAL:
[`phd_files/qualificacao-doutorado-mario-carvalho.pdf`](phd_files/qualificacao-doutorado-mario-carvalho.pdf).

```bibtex
@phdthesis{Carvalho2025qualificacao,
  author     = {Mário de Araújo Carvalho and Wesley Nunes Gonçalves},
  title      = {Deep Active Learning for Precision Agriculture: A Computational Approach},
  school     = {Universidade Federal de Mato Grosso do Sul, Faculdade de Computação},
  year       = {2025},
  month      = {mar},
  type       = {Qualificação de Doutorado},
  address    = {Campo Grande, MS, Brazil},
  note       = {PhD Qualification Examination},
  supervisor = {Wesley Nunes Gonçalves}
}
```

## Contact

- Author: Mário de Araújo Carvalho
- Email: mariodearaujocarvalho@gmail.com
- Advisor: Wesley Nunes Gonçalves (UFMS)
- Project: [https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python](https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python)

## Acknowledgements

This project is based on code from
[DeepAL: Deep Active Learning in Python](https://github.com/ej0cl6/deep-active-learning).
Please consider citing the corresponding publication, available
[here](https://arxiv.org/abs/2111.15258).

### References

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009

[4] Deep Bayesian Active Learning with Image Data, ICML, 2017

[5] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[6] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018
