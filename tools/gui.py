"""Thin shim: `poetry run python tools/gui.py` launches the DalMax inference GUI.
All logic lives in `dalmax/inference/gui.py::main()` (importable without
side effects — no window is created until `main()` runs), so this file
mirrors `trainer.py`'s pattern of a `tools/` entry point over `dalmax/`
logic.
"""

import _bootstrap  # noqa: F401  (must precede the `dalmax` import)

from dalmax.inference.gui import main

if __name__ == "__main__":
    main()
