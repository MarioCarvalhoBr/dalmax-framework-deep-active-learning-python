"""End-to-end regression test: re-run demo.py on the micro dataset and compare
against the committed golden-run fixtures in tests/golden/.

This is the closest thing to a true smoke test in this repo (see
`.specs/architecture/refactor-plan.md` Phase 1, `.specs/quality/testing-strategy.md`
"Now (Phase 1)" item 4, and `Makefile`'s `smoke` target, which runs the same CLI).

Marked `dataset` (needs the real `DATA/daninhas_full/` on disk to (re)generate
`DATA/daninhas_micro/` — see `scripts/make_micro_dataset.py`) and `slow` (each
strategy run takes several seconds of real CPU training + prediction). Not run
by `make test` / CI's fast job; run explicitly via `make test-all` or
`pytest -m dataset` on a machine with the dataset present.

Test isolation note
--------------------
This test passes `--dir_results` pointed at `tmp_path`, so only the per-run
result artifacts under that directory (`results.json`, `predictions.csv`,
`saved_model.pth`, plots) are isolated to the test's temp directory and
cleaned up automatically. `demo.py` and `utils/data.py` still write to fixed,
non-isolated repository paths as a side effect of every invocation of this
test: `results/logs/` (the run's `log-dalmax.log`, via `utils.LOGGER`),
`results/original_indices.txt` (written unconditionally by `Data.__init__` /
`create_indexes_path`), and `results/cache/` (the SSRAE feature cache and
`Y_train` pickle, via `cache_file_path` — see `.specs/quality/known-issues.md`
KI-3 and KI-21). None of these are cleaned up by this test; they accumulate in
the real `results/` tree exactly as a real `demo.py` run would.

Determinism note
-----------------
Both fixtures were captured after manually re-running the exact same CLI 3
times on the local dev notebook (no GPU): the initial labeled indices, the
queried indices, and every metric in `results.json` were bit-identical across
all 3 runs for both `RandomSampling` and `SSRAEKmeansSampling` (the latter
checked with both a cold and a warm SSRAE feature cache). No CPU-training
nondeterminism was observed for this 1-epoch/CPU/micro-dataset configuration,
so this test compares indices exactly and metrics within `1e-6` rather than
falling back to the indices-only comparison the test plan allows for. If a
future environment (different CPU, BLAS backend, or thread count) shows
nondeterminism here, that must be recorded in
`.specs/quality/known-issues.md` and this test loosened to compare indices
only, per the original test plan in `testing-strategy.md`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.make_micro_dataset import SOURCE_ROOT, generate_micro_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
PARAMS_JSON = REPO_ROOT / "files_config" / "params_micro.json"

GOLDEN_FIXTURES = sorted(GOLDEN_DIR.glob("*.json"))

INITIAL_IDXS_RE = re.compile(r"Initial labeled idxs \(sorted\): (\[[^\]]*\])")
QUERY_IDXS_RE = re.compile(r"Round 1 query_idxs \(sorted\): (\[[^\]]*\])")


def _load_fixture(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module", autouse=True)
def _ensure_micro_dataset():
    if not SOURCE_ROOT.is_dir():
        pytest.skip(f"{SOURCE_ROOT} not present; cannot (re)generate the micro dataset")
    generate_micro_dataset()


@pytest.mark.dataset
@pytest.mark.slow
@pytest.mark.parametrize("fixture_path", GOLDEN_FIXTURES, ids=lambda p: p.stem)
def test_golden_run_reproduces_indices_and_metrics(fixture_path: Path, tmp_path: Path):
    fixture = _load_fixture(fixture_path)
    expected_results = fixture["results_json"]
    strategy_name = expected_results["strategy_name"]
    n_init_labeled = expected_results["n_init_labeled"]
    n_query = expected_results["n_query"]
    n_round = expected_results["n_round"]
    seed = expected_results["seed"]

    with open(PARAMS_JSON) as f:
        params = json.load(f)
    dataset_folder = Path(params["DANINHAS"]["data_dir"].rstrip("/")).name
    n_epoch = params["DANINHAS"]["n_epoch"]

    dir_results = tmp_path / "smoke"
    cmd = [
        sys.executable,
        "demo.py",
        "--params_json",
        str(PARAMS_JSON),
        "--dataset_name",
        "DANINHAS",
        "--strategy_name",
        strategy_name,
        "--n_init_labeled",
        str(n_init_labeled),
        "--n_query",
        str(n_query),
        "--n_round",
        str(n_round),
        "--seed",
        str(seed),
        "--dir_results",
        str(dir_results),
    ]

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    combined_output = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"demo.py exited with {proc.returncode}\nstdout/stderr:\n{combined_output}"
    )

    initial_match = INITIAL_IDXS_RE.search(combined_output)
    query_match = QUERY_IDXS_RE.search(combined_output)
    assert initial_match, f"could not find initial labeled idxs in output:\n{combined_output}"
    assert query_match, f"could not find round 1 query idxs in output:\n{combined_output}"

    actual_initial_idxs = json.loads(initial_match.group(1))
    actual_query_idxs = json.loads(query_match.group(1))

    assert actual_initial_idxs == fixture["initial_labeled_idxs_sorted"]
    assert actual_query_idxs == fixture["round_1_query_idxs_sorted"]

    leaf_dir = (
        dir_results
        / dataset_folder
        / f"SEED_{seed}"
        / f"NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}"
        / strategy_name
    )
    results_json_path = leaf_dir / "results.json"
    assert results_json_path.exists(), f"expected results.json at {results_json_path}"
    with open(results_json_path) as f:
        actual_results = json.load(f)

    for key in ("all_acc", "all_precision", "all_recall", "all_f1_score"):
        assert key in actual_results, f"missing metrics key {key!r} in results.json"
        expected_values = expected_results[key]
        actual_values = actual_results[key]
        assert len(actual_values) == len(expected_values)
        for actual_v, expected_v in zip(actual_values, expected_values, strict=True):
            assert actual_v == pytest.approx(expected_v, abs=1e-6)
