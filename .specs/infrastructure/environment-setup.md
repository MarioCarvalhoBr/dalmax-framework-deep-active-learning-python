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

## Lab machine / Colab — Poetry, same as local (pip fallback retired)

**Amendment (2026-08-23)**: the pip/`requirements.txt` fallback documented below (historical) was
retired. Poetry is now the **only** supported install path on every machine — dev notebook, lab
machine, and Colab alike (see `.specs/adr/0001-adopt-poetry.md`'s amendment). The historical
`requirements.txt` file was deleted from the repo and the `export-reqs` Makefile target retired.

Lab machine / Colab install procedure (identical to the local dev notebook procedure above):
```
pipx install poetry   # once per machine, if Poetry isn't already installed
poetry install
```

### Historical record: the pre-2026-08-23 pip fallback

Poetry was previously the source of truth for dependency versions, but the lab machine
and Colab installed via a **plain pip requirements file exported from
Poetry**, not via Poetry itself:

```
poetry export -f requirements.txt --output requirements.txt --without-hashes   # historical, retired
```

(Historical — was the `export-reqs` Makefile target; required the `poetry-plugin-export` plugin.) The
`requirements.txt` file was historical — it used to sit at the repo root and started life as a
**hand-maintained pre-Poetry file**:

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
`pip install -r requirements.txt` (historical command, no longer valid) followed by `python trainer.py ...` would raise
`ModuleNotFoundError: pandas` the first time `predictions_df = pd.DataFrame(...)` executes (end of
a full run, after training — i.e. the failure surfaces late, wasting a full training run's
compute). **Status: resolved, then moot** — `pyproject.toml` declares `pandas==2.2.3`, and the
`requirements.txt` export mechanism this whole section describes was retired outright on
2026-08-23, so this gap can no longer recur by construction; see `.specs/quality/known-issues.md`
KI-2.

## CUDA note

The current (pre-refactor) `README.md` documents `CUDA==12.4` and shows a
conda install line:
```
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```
This is the CUDA toolkit version the lab machine's GPU training is expected
to target — `torch==2.5.0`/`torchvision==0.20.0` are consistent with the
historical `requirements.txt` pins above (that file no longer exists — see the
Amendment note at the top of this file). The current README also references Python
3.9 for the conda/venv setup instructions, which **predates and conflicts
with** the Poetry `python = ">=3.10,<3.13"` constraint being introduced —
this reconciliation (drop the stale 3.9 guidance) is the responsibility of
the root-docs README rewrite (a different work batch), not this file, but
is noted here since it affects what "environment setup" means going
forward: **Python 3.10–3.12** is the supported range under Poetry, not 3.9.
