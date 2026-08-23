# Environment setup

## Local dev notebook — Poetry (primary, this batch does not create `pyproject.toml`)

The Poetry environment itself (`pyproject.toml`, `poetry.toml`) is a
separate deliverable owned by a different work batch (master prompt §3).
This file documents the setup **procedure** for reference:

1. `poetry install` — installs into an in-project `.venv/` (per
   `poetry.toml`'s `[virtualenvs] in-project = true`, and per this
   environment's global `poetry config virtualenvs.in-project` already
   being `true`).
2. Python constraint: `>=3.10,<3.13` (bounded above by `torch==2.5.0`'s
   supported range; this machine runs Python 3.12.3, Poetry 2.2.1).
3. If `torch`/`torchvision` installation is too heavy for a quick local
   iteration, `poetry install --no-root` is the documented fallback (per
   the Poetry-deliverable batch) — CPU-only local usage does not need CUDA
   wheels, but standard PyPI `torch==2.5.0` wheels already include a CPU
   fallback path, so no separate CPU-only index is required locally.

## Lab machine / Colab — pip fallback from exported requirements

Poetry is the source of truth for dependency versions, but the lab machine
and Colab install via a **plain pip requirements file exported from
Poetry**, not via Poetry itself (Poetry is a local dev convenience, not
assumed present on the lab machine):

```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

(Documented as the `export-reqs` Makefile target in the Poetry-deliverable
batch; requires the `poetry-plugin-export` plugin.) The committed
`requirements.txt` currently at the repo root is a **hand-maintained
pre-Poetry file**:

```
matplotlib==3.9.2
numpy==2.1.3
Pillow==11.0.0
scikit_learn==1.5.2
seaborn==0.13.2
torch==2.5.0
torchvision==0.20.0
tqdm==4.67.1
```

**Missing `pandas` (historical, resolved in Phase 1 — KI-2)** — imported by `demo.py`
(`import pandas as pd`, used for `predictions.csv` export) and by every script in
`dalmax/reporting/` (was `utils/report/`) (`import pandas as pd`); was absent from the
hand-maintained `requirements.txt` snapshot shown above. This was a real bug at the time: a fresh
`pip install -r requirements.txt` followed by `python demo.py ...` would raise
`ModuleNotFoundError: pandas` the first time `predictions_df = pd.DataFrame(...)` executes (end of
a full run, after training — i.e. the failure surfaces late, wasting a full training run's
compute). **Status: resolved** — `pyproject.toml` declares `pandas==2.2.3` and `requirements.txt`
has since been re-exported to include it; see `.specs/quality/known-issues.md` KI-2.

Lab machine / Colab install procedure:
```
pip install -r requirements.txt      # exported-from-Poetry version, once available
```

## CUDA note

The current (pre-refactor) `README.md` documents `CUDA==12.4` and shows a
conda install line:
```
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```
This is the CUDA toolkit version the lab machine's GPU training is expected
to target — `torch==2.5.0`/`torchvision==0.20.0` are consistent with the
`requirements.txt` pins above. The current README also references Python
3.9 for the conda/venv setup instructions, which **predates and conflicts
with** the Poetry `python = ">=3.10,<3.13"` constraint being introduced —
this reconciliation (drop the stale 3.9 guidance) is the responsibility of
the root-docs README rewrite (a different work batch), not this file, but
is noted here since it affects what "environment setup" means going
forward: **Python 3.10–3.12** is the supported range under Poetry, not 3.9.
