# ADR 0001: Adopt Poetry for dependency management

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

The project previously (historical, pre-Poetry) depended on a flat `requirements.txt` (`matplotlib==3.9.2`, `numpy==2.1.3`,
`Pillow==11.0.0`, `scikit_learn==1.5.2`, `seaborn==0.13.2`, `torch==2.5.0`, `torchvision==0.20.0`,
`tqdm==4.67.1`) with no lockfile, no dependency graph resolution, and a confirmed gap: `pandas` is
imported by `demo.py` and `utils/report/*.py` but was never listed in `requirements.txt`.

The lab operates across three environments (local no-GPU dev notebook, a 2×10 GB GPU lab machine
reached via `git pull`, and Google Colab Pro for one-off runs — see
`.specs/infrastructure/execution-environments.md`), which makes reproducible, resolvable
dependency pinning more valuable than for a single-machine project. Python 3.12.3 and Poetry 2.2.1
are already installed on the local dev machine.

## Decision

We will manage dependencies with Poetry: `pyproject.toml` (`[tool.poetry]` metadata, main
dependencies including the missing `pandas`, a `dev` group for `ruff`/`pytest`/`pytest-cov`/
`pre-commit`, and `[tool.ruff]` config) plus `poetry.toml` pinning the virtualenv to `./.venv`
(`in-project = true`). At the time of this decision, `requirements.txt` was kept as a **generated
export** (`make export-reqs` → historical command, retired 2026-08-23: `poetry export -f
requirements.txt --output requirements.txt --without-hashes`), historical, for the lab machine and
Colab, which did not run Poetry directly — see this ADR's amendment below for why that export was
retired.

## Consequences

- Positive: reproducible builds via `poetry.lock`; `pandas` gap is fixed at the `pyproject.toml`
  level; one command (`poetry install`) sets up the local dev environment.
- Positive (historical — see amendment): `requirements.txt` stayed a valid, always-in-sync artifact
  for the lab machine/Colab without hand-editing two dependency lists.
- Negative (historical — `make export-reqs` itself was retired, see amendment): contributors had to
  remember to run it after any `pyproject.toml` change, or the lab machine/Colab requirements
  drift — flagged in `.specs/quality/known-issues.md` as a process risk to watch, not a code defect.
- Negative: `torch`/`torchvision` installs are heavy; Poetry resolution time on first `poetry lock`
  can be slow on the no-GPU dev machine — acceptable trade-off given reproducibility gains.

## Amendment (2026-08-23)

Amendment: `requirements.txt` export retired; Poetry is the only supported install path on all
machines (dev, lab, Colab). The historical pip fallback described above (`make export-reqs`,
`poetry export -f requirements.txt --output requirements.txt --without-hashes`) is retired — the
`requirements.txt` file was deleted from the repo, the `export-reqs` Makefile target was retired
with it, and every doc/skill/command that historically instructed `pip install -r
requirements.txt` was rewritten to `poetry install` + `poetry run ...` (see `chore/poetry-only`).
The "Negative" bullet above about remembering to re-export is retroactively moot, historical
context only — there is nothing left to export.
