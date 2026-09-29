# tools/

Command-line entry points (thin scripts over the `dalmax/` package). Run them
from the repo root; each one puts the repo root on `sys.path` itself
(`tools/_bootstrap.py`), so the working directory does not matter.

| Script | Purpose |
|---|---|
| `trainer.py` | Train/run one active-learning experiment (`dalmax.cli.main`); the campaign runner (`python -m dalmax.campaign`) calls it once per job. |
| `loader.py` | Inspect a `saved_model.pth` (`dalmax-checkpoint`): format, architecture, labels, provenance, parameter counts, CPU dummy forward. |
| `predict.py` | Run a checkpoint on one image (`--image`) or a folder (`--dir`, optional `--out`). |
| `gui.py` | tkinter mini-app for interactive predictions. |

```bash
poetry run python tools/trainer.py --help
poetry run python tools/loader.py --model results/campaign/.../saved_model.pth
poetry run python tools/predict.py --model results/campaign/.../saved_model.pth --dir DATA/daninhas_full/test/DATASET_GRAMINEA --out results/predictions/
poetry run python tools/gui.py
```

`scripts/` holds the non-CLI helpers (`make_micro_dataset.py`,
`campaign/build_manifest.py`, `colab/setup_colab.sh`). See ADR 0010.
