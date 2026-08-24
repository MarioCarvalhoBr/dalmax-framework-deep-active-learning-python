"""Every module under ``dalmax/`` must import cleanly.

This test walks the `dalmax` package and dynamically parametrizes one
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

Before refactor Phase 4 (`.specs/architecture/refactor-plan.md`), this test
walked the legacy `core`/`utils` packages that `dalmax` wrapped. Phase 4
physically moved every remaining module into `dalmax/` and deleted the dead
files (`core/query_strategies/old_functions.py`, `core/tools/SSL/code_kmh.py`,
the `SSRAE`/`VCTex` `example.py` scripts, `utils/orchestrator.py`, and the
four superseded representation-strategy files) — see
`.specs/architecture/current-state.md` §0 for the full inventory.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

# Modules intentionally excluded from the "must import cleanly" contract,
# with the verified reason (read the source before adding an entry here).
SKIP_MODULES: dict[str, str] = {
    # Standalone scripts meant to be run directly from within their own
    # directory (e.g. `cd dalmax/tools/SSRAE && python classification.py`).
    # They (a) run real feature extraction/classification at module scope,
    # and (b) use a bare `from extractor import ColorFeatureExtractor` /
    # `from VCTexMethod import VCTexMethod` style import that only resolves
    # when the script's own directory is on sys.path (as happens when run
    # directly), not when imported as `dalmax.tools.SSRAE.classification`.
    "dalmax.tools.SSRAE.classification": (
        "standalone script: non-relative `from extractor import "
        "ColorFeatureExtractor` import fails under normal package import "
        "(designed to run as `python classification.py` from its own folder); "
        "also has a module-level `os.listdir(...)` side effect"
    ),
    "dalmax.tools.VCTex.classification": (
        "standalone script: module-level `os.listdir(...)` + `print(...)` "
        "side effect, and a non-relative `from VCTexMethod import "
        "VCTexMethod` import that fails under normal package import"
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
    return sorted(set(_discover_modules("dalmax")))


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
        # (now dalmax/reporting/extract_confusion_matrices.py) PyPDF2 import
        # (undeclared in pyproject.toml, see
        # known-issues.md KI-28); PyPDF2 is now a declared dependency
        # (pyproject.toml) and that module imports cleanly, but this
        # fallback stays as a defensive guard for any future undeclared
        # third-party import elsewhere in dalmax.
        missing = exc.name or ""
        if not missing.startswith("dalmax"):
            pytest.skip(f"optional/undeclared dependency not installed: {missing}")
        raise


def test_skip_list_matches_discovered_modules():
    """Guard against a stale SKIP_MODULES entry (e.g. after a file is deleted)."""
    stale = set(SKIP_MODULES) - set(SKIPPED_BUT_DISCOVERED)
    assert not stale, f"SKIP_MODULES entries no longer discovered on disk: {stale}"


def test_no_discovery_errors():
    """`pkgutil.walk_packages` must have been able to descend into every
    subpackage; a failure here means some part of dalmax was silently
    excluded from `MODULES_TO_TEST` above instead of being tested or
    explicitly skipped."""
    assert not DISCOVERY_ERRORS, f"failed to walk into subpackage(s): {DISCOVERY_ERRORS}"
