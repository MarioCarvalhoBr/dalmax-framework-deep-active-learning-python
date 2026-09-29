# Data safety

## Immutability and append-only policy

- **`DATA/` is immutable.** `DATA/daninhas_full/{train,test}/DATASET_{BRACHIARIA,
  COLONIAO,GRAMINEA,MAMONA,OUTRAS_FOLHAS_LARGAS}/` (~10,193 files, ~47 MB) and
  `DATA/DATA_CIFAR10/` must never be edited, moved, renamed, or deleted by any
  agent or command. No code change should write into `DATA/`.
- **`results/` is append-only.** Never edit or delete a past run's directory
  (`results/<run>/<dataset>/SEED_*/NQ_*_NIL_*_NR_*_NE_*/<strategy>/`). Reporting
  scripts (`dalmax/reporting/*.py`, was `utils/report/*.py`) may read from `results/` and write new derived
  files (CSVs, averaged plots) but must not overwrite or remove existing raw
  `results.json` / `predictions.csv` / plot files from prior runs.
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
