# Data safety

## Immutability and append-only policy

- **`DATA/` is immutable.** `DATA/daninhas_full/{train,test}/DATASET_{BRACHIARIA,
  COLONIAO,GRAMINEA,MAMONA,OUTRAS_FOLHAS_LARGAS}/` (10,193 files, ~47 MB; the dataset of the
  first campaign, with duplicates, KI-46), `DATA/daninhas_full_fixed/` (the canonical dataset
  since 2026-11-11, 8,240 files) and `DATA/DATA_CIFAR10/` must never be edited, moved, renamed,
  or deleted by any agent or command. No code change should write into an existing `DATA/`
  directory.
- **One sanctioned way to CREATE a dataset directory (author instruction 2026-11-11, ADR 0016):
  `scripts/datasets/build_daninhas_full_fixed.py`.** It reads `DATA/daninhas_full/` only, writes a
  NEW directory (`DATA/daninhas_full_fixed/` by default), refuses to run when the destination
  exists, never modifies an existing dataset directory, and verifies its result (`--check`: no
  repeated MD5 inside or across the splits; read-only, also run by `make dataset-check` and by
  `scripts/colab/setup_colab.sh` after unzipping). Once built, `daninhas_full_fixed` is immutable
  like every other dataset: a different dataset is a new directory with a new name, never a
  rebuild in place. `scripts/make_micro_dataset.py` (`DATA/daninhas_micro/`, the golden-test
  fixture) is the other generator; it never writes into its source either.
- **`results/` is append-only.** Never edit, move or delete a past run's directory
  (`results/<run>/<dataset>/SEED_*/NQ_*_NIL_*_NR_*_NE_*/<strategy>/`). Reporting
  scripts (`dalmax/reporting/*.py`, was `utils/report/*.py`) may read from `results/` and write new derived
  files (CSVs, averaged plots) but must not overwrite or remove existing raw
  `results.json` / `predictions.csv` / plot files from prior runs. New experiments go to NEW
  results roots (e.g. `results/campaign_fixed/`, `results/campaign_kmh_fixed/`, ADR 0016) rather
  than into the leaves of an executed tree. There is no exception: the ADR 0014 archive script
  (`scripts/campaign/archive_superseded_runs.py`, which moved superseded leaves to
  `results/old_results_broken/`) was deleted unexecuted on 2026-11-11 (ADR 0016), and no code may move
  or delete anything under `results/`.
- **`phd_files/` is read-only context** (the qualification PDF, the paper under
  revision, the LaTeX sources). Agents may read `phd_files/Active_Learning_Mario/
  method_full.tex` for notation but must never write there. The only paper-related
  write location is `paper_drafts/` (owned by the `paper-liaison` agent).

## No secrets, no personal paths

- No dataset images, credentials, API keys, or `.env` contents may be committed.
- No absolute personal filesystem paths (e.g.
  `/home/carvalho/Desktop/UFMS/...`) in committed files — use paths relative to
  the repo root instead.

## Large artifacts

- Large artifacts (>5 MB), and any `.pkl` or `.pth` file, must never be committed.
  This includes the embedding caches (`results/features_dict_ssrae.pkl`,
  `results/features_dict_vctex.pkl`, `results/Y_train.pkl`) and any saved model
  checkpoint written under a results directory. `.gitignore` must keep `*.pkl` and
  `*.pth` excluded.

## Enforcement

- `settings.json` denies `Edit`/`Write` under `DATA/**`, `phd_files/**`, and
  `results/**` at the tool-permission level, and denies `Bash(rm -rf *)`.
- The `experiment-auditor` and `code-reviewer` agents both check for accidental
  writes into these paths before approving a change.
