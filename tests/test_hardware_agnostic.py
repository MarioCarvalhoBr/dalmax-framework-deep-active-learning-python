"""Hardware-agnostic gate (ADR 0010, `.claude/rules/reproducibility.md`).

No GPU model name may appear in any tracked file: the GPU actually used is
recorded only at runtime in each run's `run_metadata.json` (`environment.gpus`).
The only tolerated hits are the past-tense execution-history records listed in
`HISTORY_ALLOWLIST` (what the 2026-08-25/26 ablation batch actually ran on).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Built from pieces so this file never matches its own pattern.
_MODELS = ("a" + "100", "t" + "4", "v" + "100", "tes" + "la")
PATTERN = re.compile(rf"(?<![0-9A-Za-z])(?:{'|'.join(_MODELS)})(?![0-9A-Za-z])", re.IGNORECASE)

EXCLUDED_PREFIXES = ("phd_files/", "paper_drafts/", "docs/results/", "results/", "DATA/")
EXCLUDED_FILES = {"poetry.lock", "tests/test_hardware_agnostic.py"}

# Past-tense execution-history records (which hardware a finished batch ran on).
HISTORY_ALLOWLIST = {
    ".specs/architecture/refactor-plan.md",
    ".specs/experiments/ablation-study.md",
    ".specs/experiments/baseline-results.md",
    ".specs/infrastructure/execution-environments.md",
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def test_no_gpu_model_names_outside_history_records() -> None:
    offenders: list[str] = []
    for rel in _tracked_files():
        if rel in EXCLUDED_FILES or rel in HISTORY_ALLOWLIST or rel.startswith(EXCLUDED_PREFIXES):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary artifact
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not offenders, "GPU model name(s) found:\n" + "\n".join(offenders)


def test_history_allowlist_only_names_existing_files() -> None:
    for rel in HISTORY_ALLOWLIST:
        assert (REPO_ROOT / rel).is_file(), rel


def test_campaign_has_no_hardware_specific_names() -> None:
    from dalmax import campaign as C

    manifest = C.load_manifest(REPO_ROOT / C.DEFAULT_MANIFEST)
    assert manifest.results_root == "results/campaign"
    assert manifest.name == "campaign"
