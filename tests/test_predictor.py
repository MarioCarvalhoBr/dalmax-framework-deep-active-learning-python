"""Tests for `dalmax.inference.predictor.Predictor`.

Uses a random-initialized (untrained) `DaninhasModelResNet50` -- no training
loop needed to exercise the predictor's I/O contract (well-formed
CSV/JSON-serializable rows, probabilities summing to 1, determinism). Marked
`dataset` because it reads 3 real images from `DATA/daninhas_micro/` (see
`scripts/make_micro_dataset.py`) to exercise the exact preprocessing path
(`PIL` decode -> resize -> handler transform) end to end, rather than a
synthetic tensor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dalmax.inference.predictor import Predictor
from dalmax.models.checkpoint import save_checkpoint
from dalmax.models.daninhas_resnet50 import DaninhasModelResNet50

REPO_ROOT = Path(__file__).resolve().parent.parent
MICRO_TEST_DIR = REPO_ROOT / "DATA" / "daninhas_micro" / "test"

CLASS_NAMES = [
    "DATASET_BRACHIARIA",
    "DATASET_COLONIAO",
    "DATASET_GRAMINEA",
    "DATASET_MAMONA",
    "DATASET_OUTRAS_FOLHAS_LARGAS",
]
IMG_SIZE = 128


@pytest.fixture(scope="module")
def sample_image_paths() -> list[Path]:
    if not MICRO_TEST_DIR.is_dir():
        pytest.skip(f"{MICRO_TEST_DIR} not present; run scripts/make_micro_dataset.py first")
    paths = sorted(MICRO_TEST_DIR.glob("*/*.jpg"))[:3]
    if len(paths) < 3:
        pytest.skip(f"fewer than 3 images found under {MICRO_TEST_DIR}")
    return paths


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory) -> str:
    clf = DaninhasModelResNet50(n_classes=len(CLASS_NAMES))
    clf.eval()
    path = str(tmp_path_factory.mktemp("checkpoints") / "saved_model.pth")
    save_checkpoint(
        path,
        clf,
        model_name="DANINHAS",
        n_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
        extra={"strategy_name": "RandomSampling", "seed": 1, "dataset_name": "DANINHAS"},
    )
    return path


@pytest.mark.dataset
def test_predictor_loads_checkpoint_metadata(checkpoint_path):
    predictor = Predictor(checkpoint_path, device="cpu")
    assert predictor.class_names == CLASS_NAMES
    assert predictor.img_size == IMG_SIZE
    assert predictor.dataset_name == "DANINHAS"
    assert predictor.extra["strategy_name"] == "RandomSampling"


@pytest.mark.dataset
def test_predict_paths_well_formed(checkpoint_path, sample_image_paths):
    predictor = Predictor(checkpoint_path, device="cpu")
    rows = predictor.predict_paths(sample_image_paths)

    assert len(rows) == len(sample_image_paths)
    for row, path in zip(rows, sample_image_paths, strict=True):
        assert row.path == str(path)
        assert row.predicted_class in CLASS_NAMES
        assert set(row.probabilities) == set(CLASS_NAMES)

        # Probabilities are a valid distribution over all classes.
        prob_sum = sum(row.probabilities.values())
        assert prob_sum == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= row.confidence <= 1.0

        # confidence is exactly the argmax probability, and predicted_class
        # is exactly the argmax class.
        best_class = max(row.probabilities, key=row.probabilities.get)
        assert row.predicted_class == best_class
        assert row.confidence == pytest.approx(row.probabilities[best_class], abs=1e-9)


@pytest.mark.dataset
def test_predict_paths_is_deterministic(checkpoint_path, sample_image_paths):
    predictor_a = Predictor(checkpoint_path, device="cpu")
    predictor_b = Predictor(checkpoint_path, device="cpu")

    rows_a = predictor_a.predict_paths(sample_image_paths)
    rows_b = predictor_b.predict_paths(sample_image_paths)

    assert [r.predicted_class for r in rows_a] == [r.predicted_class for r in rows_b]
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        assert row_a.probabilities == pytest.approx(row_b.probabilities, abs=1e-9)

    # Running the same predictor twice is also deterministic (eval mode, no
    # dropout/batchnorm-update side effects across calls).
    rows_a_again = predictor_a.predict_paths(sample_image_paths)
    for row_a, row_a2 in zip(rows_a, rows_a_again, strict=True):
        assert row_a.probabilities == pytest.approx(row_a2.probabilities, abs=1e-9)


@pytest.mark.dataset
def test_predict_paths_empty_list(checkpoint_path):
    predictor = Predictor(checkpoint_path, device="cpu")
    assert predictor.predict_paths([]) == []
