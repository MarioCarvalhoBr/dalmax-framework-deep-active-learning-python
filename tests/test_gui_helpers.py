"""Headless-CI-safe tests for `dalmax.inference.gui`.

`dalmax.inference.gui` must be importable without a display (no `tkinter`
window is constructed at import time -- all of that lives inside `main()`,
see that module's docstring). This test only imports the module and
exercises its plain-Python helper functions, never calling `main()`.
"""

from __future__ import annotations

from dalmax.inference.gui import (
    collect_image_paths,
    format_confidence_percent,
    format_model_summary,
    format_probabilities,
)
from dalmax.inference.predictor import PredictionRow


def test_import_is_side_effect_free():
    """Importing the module (done implicitly by every other test in this
    file) must not require a display or create any tkinter object -- if it
    did, collecting this test file in headless CI would already fail."""
    import dalmax.inference.gui as gui_module

    assert hasattr(gui_module, "main")


def test_collect_image_paths_finds_images_recursively(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.PNG").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_bytes(b"fake")

    found = collect_image_paths(tmp_path)

    assert found == sorted(found)
    assert {p.name for p in found} == {"a.jpg", "b.PNG"}


def test_collect_image_paths_empty_dir(tmp_path):
    assert collect_image_paths(tmp_path) == []


def test_format_confidence_percent():
    assert format_confidence_percent(0.8734) == "87.34%"
    assert format_confidence_percent(1.0) == "100.00%"
    assert format_confidence_percent(0.0) == "0.00%"


def test_format_model_summary():
    class _FakePredictor:
        class_names = ["A", "B", "C"]
        img_size = 128
        dataset_name = "DANINHAS"
        extra = {"strategy_name": "RandomSampling", "dataset_name": "DANINHAS", "seed": 1}

    summary = format_model_summary(_FakePredictor())
    assert "RandomSampling" in summary
    assert "DANINHAS" in summary
    assert "A, B, C" in summary
    assert "128x128" in summary


def test_format_model_summary_missing_extra_falls_back():
    class _FakePredictor:
        class_names = ["A", "B"]
        img_size = 32
        dataset_name = "CIFAR10"
        extra: dict = {}

    summary = format_model_summary(_FakePredictor())
    assert "unknown" in summary
    assert "CIFAR10" in summary


def test_format_probabilities_sorted_descending():
    row = PredictionRow(
        path="p.jpg",
        predicted_class="B",
        confidence=0.7,
        probabilities={"A": 0.1, "B": 0.7, "C": 0.2},
    )
    formatted = format_probabilities(row)
    lines = formatted.split("\n")
    assert lines[0].startswith("B: 70.00%")
    assert lines[1].startswith("C: 20.00%")
    assert lines[2].startswith("A: 10.00%")
