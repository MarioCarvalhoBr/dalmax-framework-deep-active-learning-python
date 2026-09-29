"""Training entry point: `python tools/trainer.py ...` (moved from the repo
root to `tools/` on 2026-09-29, ADR 0010; used by `dalmax.campaign`, the
`Makefile` targets and `tests/test_golden_run.py`). A thin shim: all actual
argparse/config/run/report logic lives in `dalmax.cli`
(`.specs/architecture/target-architecture.md` §10, `.claude/rules/code-quality.md`
"single responsibility" — the old 289-line `main()` used to mix CLI parsing,
the training loop, plotting, and persistence in this one file).

Renamed from the historical `demo.py` to `trainer.py` on 2026-08-23 (`git mv`, see ADR
0006) — a pure rename, no behavior change: the historical name `demo.py` no
longer exists anywhere in this repository. This file trains a model and
writes `saved_model.pth`; the companion inference tools that consume that
checkpoint (`predict.py`, `loader.py`, `gui.py`, `dalmax/inference/`) live
alongside it in `tools/`.
"""

import _bootstrap  # noqa: F401  (must precede the `dalmax` import)

from dalmax.cli import main

if __name__ == "__main__":
    main()
