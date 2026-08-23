"""Every module under ``core/`` and ``utils/`` must import cleanly.

This test walks both packages and dynamically parametrizes one
``importlib.import_module`` check per discovered module. A handful of modules
are excluded up front (see ``SKIP_MODULES`` below) because they are not
meant to be imported as part of the package: they are standalone scripts
that execute real work (feature extraction, training, plotting) at module
scope, or use import styles that only work when run directly (e.g.
``python classification.py`` from inside their own folder).

Everything else is expected to import without executing GPU code, touching
DATA/, or running training. This was verified by reading each source file:
no module outside the skip list calls ``torch.cuda`` (or similar) at module
scope, all such calls live inside function/method bodies and only run when
those functions are called.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

# Modules intentionally excluded from the "must import cleanly" contract,
# with the verified reason (read the source before adding an entry here).
SKIP_MODULES: dict[str, str] = {
    # Entirely a triple-quoted string literal (dead code kept for reference,
    # see .specs/quality/known-issues.md). It technically imports without
    # raising, but it is explicitly called out in the harness spec as a file
    # to skip: it is not real, executable module content.
    "core.query_strategies.old_functions": (
        "dead/experimental file (body is a single docstring, see known-issues.md); "
        "explicitly excluded per harness spec"
    ),
    # Example/demo scripts meant to be run directly from within their own
    # directory (e.g. `cd core/tools/SSRAE && python example.py`). They both
    # (a) run real feature extraction on a random image at module scope, and
    # (b) use a bare `from extractor import ColorFeatureExtractor` style
    # import that only resolves when the script's own directory is on
    # sys.path (as happens when run as `python example.py`), not when
    # imported as `core.tools.SSRAE.example`.
    "core.tools.SSRAE.example": (
        "standalone demo script: runs feature extraction at import time and "
        "uses a non-relative `from extractor import ...` import that fails "
        "under normal package import"
    ),
    "core.tools.VCTex.example": (
        "standalone demo script: runs feature extraction at import time and "
        "uses a non-relative `from VCTexMethod import ...`-style import that "
        "fails under normal package import"
    ),
    # Same non-relative-import problem, plus a module-level `os.listdir` +
    # `print` side effect executed unconditionally.
    "core.tools.SSRAE.classification": (
        "standalone script: non-relative `from extractor import "
        "ColorFeatureExtractor` import fails under normal package import "
        "(designed to run as `python classification.py` from its own folder)"
    ),
    "core.tools.VCTex.classification": (
        "standalone script: module-level `os.listdir(...)` + `print(...)` "
        "side effect, and a non-relative `from VCTexMethod import "
        "VCTexMethod` import that fails under normal package import"
    ),
    # This is the most severe offender: an exploratory script, not a module.
    # At import time it unconditionally (a) loads
    # results/features_dict_ssrae.pkl (~130 MB on this machine) and
    # results/Y_train.pkl, calling `exit()` if either is missing (so a fresh
    # clone without a prior experiment run would raise SystemExit on
    # import); (b) runs a full sklearn TSNE.fit_transform over the *entire*
    # cached feature set (confirmed to take minutes of 100%+ CPU — this is
    # what made the first `pytest` run of this suite appear to hang); (c)
    # builds `torch.tensor(..., device="cuda", ...)` unconditionally, which
    # raises on any machine without a GPU; and (d) ends with `plt.show()`,
    # which blocks indefinitely waiting for a GUI event loop on any machine
    # with a display/GUI matplotlib backend (confirmed: this machine's
    # default backend is TkAgg with DISPLAY set). No `if __name__ ==
    # "__main__":` guard exists anywhere in the file.
    "core.tools.SSL.code_kmh": (
        "exploratory script, not an importable module: unconditionally loads "
        "large results/*.pkl caches (or exit()s if absent), runs a full "
        "TSNE.fit_transform at import time (minutes of CPU), requires CUDA, "
        "and ends with a blocking plt.show()"
    ),
}


# Populated by `_discover_modules` when `pkgutil.walk_packages` cannot even
# descend into a subpackage (i.e. importing an intermediate __init__.py
# raises). `pkgutil` silently swallows ImportError during traversal unless
# given an `onerror` callback, which would otherwise hide whole subtrees of
# modules from this test instead of failing loudly.
DISCOVERY_ERRORS: dict[str, str] = {}


def _discover_modules(package_name: str) -> list[str]:
    def _on_error(name: str) -> None:
        DISCOVERY_ERRORS[name] = repr(sys.exc_info()[1])

    package = importlib.import_module(package_name)
    names = [package_name]
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=f"{package_name}.", onerror=_on_error
    ):
        if "__pycache__" in module_info.name:
            continue
        names.append(module_info.name)
    return names


def _all_module_names() -> list[str]:
    names: list[str] = []
    for top in ("core", "utils"):
        names.extend(_discover_modules(top))
    return sorted(set(names))


ALL_MODULES = _all_module_names()
MODULES_TO_TEST = [name for name in ALL_MODULES if name not in SKIP_MODULES]
SKIPPED_BUT_DISCOVERED = [name for name in ALL_MODULES if name in SKIP_MODULES]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module_imports_cleanly(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # A missing *third-party* dependency (not one of our own modules) is
        # an environment/packaging gap, not a code defect this test should
        # fail on. Historically this covered utils/report/1_cm_extract_from_pdf.py's
        # PyPDF2 import (undeclared in requirements.txt/pyproject.toml, see
        # known-issues.md KI-28); PyPDF2 is now a declared dependency
        # (pyproject.toml) and that module imports cleanly, but this
        # fallback stays as a defensive guard for any future undeclared
        # third-party import elsewhere in core/utils.
        missing = exc.name or ""
        if not missing.startswith(("core", "utils")):
            pytest.skip(f"optional/undeclared dependency not installed: {missing}")
        raise


def test_skip_list_matches_discovered_modules():
    """Guard against a stale SKIP_MODULES entry (e.g. after a file is deleted)."""
    stale = set(SKIP_MODULES) - set(SKIPPED_BUT_DISCOVERED)
    assert not stale, f"SKIP_MODULES entries no longer discovered on disk: {stale}"


def test_no_discovery_errors():
    """`pkgutil.walk_packages` must have been able to descend into every
    subpackage; a failure here means some part of core/utils was silently
    excluded from `MODULES_TO_TEST` above instead of being tested or
    explicitly skipped."""
    assert not DISCOVERY_ERRORS, f"failed to walk into subpackage(s): {DISCOVERY_ERRORS}"
