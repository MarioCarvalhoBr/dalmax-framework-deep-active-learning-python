"""End-to-end micro runs of the eight DeepAL+ ports (ADR 0015) through
`tools/trainer.py`: 2 query rounds on `DATA/daninhas_micro/`, CPU, seed 1.

Marked `dataset` (needs `DATA/daninhas_micro/`; this test never (re)generates it,
see `scripts/make_micro_dataset.py` / `make micro-dataset`) and `slow` (real ResNet-50
training on CPU; WAAL and VAAL take minutes). Run with
`poetry run pytest -m "dataset and slow" tests/test_deepalplus_e2e.py`.

Each run must finish, write three rounds of metrics, record the resolved
`strategy_args` in `run_metadata.json`, query disjoint unlabeled ids, and leave a
`saved_model.pth` that the inference `Predictor` loads and that reproduces the run's
own `predictions.csv` (WAAL and LossPredictionLoss train with their own trainers but
save the ordinary task model). Every port reports `strategy_diagnostics`, and no query
may be the pool-order head (the KI-45 failure). CEAL must report its pseudo-labels.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dalmax.inference.predictor import Predictor

REPO_ROOT = Path(__file__).resolve().parent.parent
MICRO_ROOT = REPO_ROOT / "DATA" / "daninhas_micro"
CAMPAIGN = REPO_ROOT / "files_config" / "campaign"

PARAMS_BY_STRATEGY = {
    "VarRatio": CAMPAIGN / "params_paper1_mc_dropout_micro.json",
    "MeanSTD": CAMPAIGN / "params_paper1_mc_dropout_micro.json",
    "KCenterGreedyPCA": CAMPAIGN / "params_paper1_micro.json",
    "BadgeSampling": CAMPAIGN / "params_paper1_micro.json",
    "CEALSampling": CAMPAIGN / "params_paper1_micro.json",
    "LossPredictionLoss": CAMPAIGN / "params_paper1_micro.json",
    "VAAL": CAMPAIGN / "params_paper1_vaal_micro.json",
    "WAAL": CAMPAIGN / "params_paper1_micro.json",
}

N_INIT, N_QUERY, N_ROUND, SEED = 10, 5, 2, 1
N_POOL_UPPER = 100_000  # any bound above the pool size: only the first 2n unlabeled ids matter
INITIAL_RE = re.compile(r"Initial labeled idxs \(sorted\): (\[[^\]]*\])")
QUERY_RE = re.compile(r"Round (\d+) query_idxs \(sorted\): (\[[^\]]*\])")


@pytest.mark.dataset
@pytest.mark.slow
@pytest.mark.parametrize("strategy_name", sorted(PARAMS_BY_STRATEGY))
def test_micro_run_end_to_end(strategy_name: str, tmp_path: Path) -> None:
    if not (MICRO_ROOT / "train").is_dir():
        pytest.skip(f"{MICRO_ROOT} not present (run `make micro-dataset`)")
    params_json = PARAMS_BY_STRATEGY[strategy_name]
    params = json.loads(params_json.read_text())["DANINHAS"]
    dir_results = tmp_path / "e2e"
    cmd = [
        sys.executable,
        "tools/trainer.py",
        "--params_json",
        str(params_json),
        "--dataset_name",
        "DANINHAS",
        "--strategy_name",
        strategy_name,
        "--n_init_labeled",
        str(N_INIT),
        "--n_query",
        str(N_QUERY),
        "--n_round",
        str(N_ROUND),
        "--seed",
        str(SEED),
        "--device",
        "cpu",
        "--dir_results",
        str(dir_results),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"trainer.py exited with {proc.returncode}:\n{output[-5000:]}"

    initial = set(json.loads(INITIAL_RE.search(output).group(1)))
    queries = {int(rd): json.loads(ids) for rd, ids in QUERY_RE.findall(output)}
    assert sorted(queries) == [1, 2]
    assert all(len(ids) == N_QUERY for ids in queries.values())
    assert not (set(queries[1]) & set(queries[2])) and not (set(queries[1]) | set(queries[2])) & initial

    leaf = (
        dir_results
        / Path(params["data_dir"].rstrip("/")).name
        / f"SEED_{SEED}"
        / f"NQ_{N_QUERY}_NIL_{N_INIT}_NR_{N_ROUND}_NE_{params['n_epoch']}"
        / strategy_name
    )
    results = json.loads((leaf / "results.json").read_text())
    assert results["rounds"] == [0, 1, 2] and len(results["all_f1_macro"]) == 3
    diag = results["strategy_diagnostics"]
    if strategy_name == "CEALSampling":
        assert diag["ceal_round"] == [1, 2] and len(diag["ceal_n_pseudo_labels"]) == 2
    # Not pool order (the KI-45 signature): the query is not drawn from the first 2n
    # unlabeled positions (~10 of ~800 here), checked both from the reported diagnostic
    # and from the logged query ids.
    assert len(diag["pool_order_fraction"]) == N_ROUND
    assert all(fraction < 0.8 for fraction in diag["pool_order_fraction"]), diag
    labeled = set(initial)
    for rd in (1, 2):
        unlabeled = [i for i in range(N_POOL_UPPER) if i not in labeled][: 2 * N_QUERY]
        assert not set(queries[rd]) <= set(unlabeled), (rd, queries[rd])
        labeled |= set(queries[rd])

    metadata = json.loads((leaf / "run_metadata.json").read_text())
    assert "strategy_args" in metadata["config"]["dataset"]

    with open(leaf / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))[:5]
    predictor = Predictor(str(leaf / "saved_model.pth"), device="cpu")
    predicted = predictor.predict_paths([row["Path"] for row in rows])
    for row, pred in zip(rows, predicted, strict=True):
        assert pred.predicted_class == row["Predicted Class"]
