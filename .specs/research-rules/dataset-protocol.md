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
everywhere downstream, e.g. in confusion matrices). Inside each class folder the files are taken in plain
lexicographic order of their names (`dalmax.data.loaders.list_image_names`, since 2026-11-12; before that an unsorted
`os.listdir`, whose order differed between machines and Colab sessions, KI-47), so the pool and test lists, and what
every pool/test index names, are the same everywhere (`dalmax/data_order.py`; `run_metadata.json` `data_order`
records their hashes). Images are opened,
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

## Canonical since 2026-11-11: `daninhas_full_fixed` (KI-46, ADR 0016)

`daninhas_full` holds byte-identical copies inside and across the splits (`.specs/quality/known-issues.md` KI-46).
The duplicate-free `DATA/daninhas_full_fixed/` (built by `scripts/datasets/build_daninhas_full_fixed.py`, same layout
and class folders, plus metadata files) is the dataset of every run from 2026-11-11 on. Counts
(measured with `find ... -type f | wc -l` on the built folder, 2026-11-11):

| Class | train | test |
|---|---|---|
| `DATASET_BRACHIARIA` | 365 | 172 |
| `DATASET_COLONIAO` | 982 | 89 |
| `DATASET_GRAMINEA` | 1,156 | 502 |
| `DATASET_MAMONA` | 3,056 | 167 |
| `DATASET_OUTRAS_FOLHAS_LARGAS` | 1,671 | 78 |
| **Total** | **7,230** | **1,008** |

Metadata layout (author instruction 2026-11-12; the top-level files describe the dataset as it is, the provenance
lives under `logs/`):

```
daninhas_full_fixed/
  train/<class>/...         7,230 images
  test/<class>/...          1,008 images
  README.txt                identity rule, counts per split and class, how to verify (no source counts)
  MANIFEST.csv              split,class,file,md5 (the current files only)
  logs/build_history.txt    builder, exclusion list, source file count, what was removed and why (the build report)
  logs/source_manifest.csv  MANIFEST.csv plus source_train_copies, source_test_copies, source_aliases_dropped
```

Total files 8,242 (8,238 images + 4 metadata files; 8,240 in the previous layout, `MANIFEST.csv` with the source
columns + `README.txt` with the source counts, converted once by the builder's `--migrate-metadata`, images
untouched). The Colab setup checks the images per split (7,230 / 1,008) and runs the builder's `--check`: the files on
disk equal the rows of `MANIFEST.csv` (name and MD5), the metadata layout, no repeated MD5. The metadata never
affects a run: the loaders, the `data_order` hashes (train `805314e2dae2`, test `17ac35b4b1b3`) and the cache keys
read only the paths under `train/` and `test/`. The tables above and below describe the original `daninhas_full`, kept as the dataset of the
first campaign.

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
  `demo.py` (historical)/`dalmax/cli.py`. The active-learning initial pool
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
