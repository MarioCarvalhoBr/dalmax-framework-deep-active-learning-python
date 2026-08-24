"""Thin shim: `poetry run python gui.py` launches the DalMax inference GUI.
All logic lives in `dalmax/inference/gui.py::main()` (importable without
side effects — no window is created until `main()` runs), so this file
mirrors `trainer.py`'s pattern of a repo-root entry point over `dalmax/`
logic.
"""

from dalmax.inference.gui import main

if __name__ == "__main__":
    main()
