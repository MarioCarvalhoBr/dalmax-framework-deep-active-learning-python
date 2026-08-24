# Dataset protocol — `daninhas_full`

## Origin

UAV (drone) imagery of weed species relevant to Brazilian precision
agriculture, pre-cropped into fixed-size RGB patches. Exact acquisition
methodology (flight altitude, sensor, patch-extraction pipeline) is **TBD**
— not documented in the code or in the files read for this batch; if needed
for the paper's dataset section, source it from
`phd_files/Active_Learning_Mario/` or the qualification document
(`phd_files/qualificacao-doutorado-mario-carvalho.pdf`), not covered here.

## Layout

```
DATA/daninhas_full/
├── train/
│   ├── DATASET_BRACHIARIA/
│   ├── DATASET_COLONIAO/
│   ├── DATASET_GRAMINEA/
│   ├── DATASET_MAMONA/
│   └── DATASET_OUTRAS_FOLHAS_LARGAS/
├── test/
│   ├── DATASET_BRACHIARIA/
│   ├── DATASET_COLONIAO/
│   ├── DATASET_GRAMINEA/
│   ├── DATASET_MAMONA/
│   └── DATASET_OUTRAS_FOLHAS_LARGAS/
└── arquivos.txt          # pre-computed per-class file counts (see below)
```

Loaded by `dalmax.data.loaders.get_DANINHAS` (was `utils.data.get_DANINHAS`, moved in Phase 4): classes are discovered as
`sorted(os.listdir(train_dir))` (i.e. the 5 `DATASET_*` folder names, sorted
alphabetically — this sort order fixes the integer class index mapping used
everywhere downstream, e.g. in confusion matrices). Images are opened,
converted to RGB, and resized to 128×128 (`img_size=128` default parameter).

## Per-class counts (measured directly, `find ... | wc -l`, confirms `arquivos.txt`)

| Class | Train | Test | Total | Train share | Test share |
|---|---|---|---|---|---|
| BRACHIARIA | 774 | 335 | 1,109 | 9.6% | 15.9% |
| COLONIAO | 1,036 | 190 | 1,226 | 12.8% | 9.0% |
| GRAMINEA | 1,198 | 970 | 2,168 | 14.8% | 46.1% |
| MAMONA | 3,370 | 456 | 3,826 | 41.7% | 21.7% |
| OUTRAS_FOLHAS_LARGAS | 1,708 | 155 | 1,863 | 21.1% | 7.4% |
| **Total** | **8,086** | **2,106** | **10,192** | 100% | 100% |

(`arquivos.txt` reports `Total: 8086` train / `Total: 2106` test,
`Average: 1617.2` train / `421.2` test per class — matches. Grand total
10,192 files; SHARED_CONTEXT's "~10,193" figure is consistent within
rounding/an off-by-one, not investigated further.)

## Class imbalance notes

- **Train/test class distributions are not proportional to each other.**
  `MAMONA` is the largest train class (41.7% of train) but only the
  second-largest test class (21.7%); `GRAMINEA` is the largest test class by
  far (46.1% of test, driven by 970 images) despite being only the
  third-largest train class (14.8%). This means a classifier's test
  performance is disproportionately sensitive to `GRAMINEA` accuracy
  relative to its training representation — a `weighted`-average metric
  (see `metrics.md`) will similarly weight `GRAMINEA` most heavily among
  test classes (46%), while a macro-average metric (needed for the
  ablation study) would not.
- Train-set imbalance ratio (largest/smallest class): `3370/774 ≈ 4.35×`
  (`MAMONA` vs `BRACHIARIA`). Test-set imbalance ratio:
  `970/155 ≈ 6.26×` (`GRAMINEA` vs `OUTRAS_FOLHAS_LARGAS`).
- No class-balancing (oversampling, class weights, stratified sampling) was
  observed in `dalmax/data/datasets.py` (was `utils/data.py`),
  `dalmax/models/daninhas_resnet50.py` (was `core/daninhas_model.py`; not read in this
  batch — TBD confirm no `class_weight`/`WeightedRandomSampler` there), or
  `demo.py`/`dalmax/cli.py`. The active-learning initial pool
  (`Data.initialize_labels`) is a uniform random draw of `n_init_labeled`
  images from the **full, imbalanced** train pool
  (`np.random.shuffle(tmp_idxs); labeled_idxs[tmp_idxs[:n_init_labeled]] = True`)
  — no class-stratified seeding.

## Secondary dataset — CIFAR10

`DATA/DATA_CIFAR10/` is used as a secondary benchmark
(`dalmax.data.loaders.get_CIFAR10`, was `utils.data.get_CIFAR10`, moved in Phase 4;
same folder-per-class loading pattern, images
resized to 32×32, 10 balanced classes by construction). Not affected by the
daninhas-specific hardcodes noted in `known-issues.md` (owned by another
batch). The `SSLStrategy` hardcoded `'DANINHAS'` params key that used to block
`SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling` from running against CIFAR10 at
all (KI-4) was resolved in Phase 2 and the file itself deleted in Phase 4; what
remains is a content gap, not a code gap — `files_config/benchmark/params_df_gpu_*.json`'s `CIFAR10`
block still has no `config_kmh`/`selection.hierarchy` entry, so those two
strategies raise a `dalmax.config.schema.ConfigError` (not a `KeyError`) against
CIFAR10 until one is added (KI-23, still open — see
`experiments/ablation-study.md` §6.2).
