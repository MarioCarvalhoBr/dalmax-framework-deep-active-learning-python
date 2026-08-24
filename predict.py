"""Run a trained `dalmax-checkpoint` (`dalmax.models.checkpoint`, written by
`trainer.py`/`Strategy.save_model`) on one image or a folder of images,
outside the active-learning loop.

Examples
--------
    poetry run python predict.py --model results/dalmax1/.../saved_model.pth \\
        --image DATA/daninhas_micro/test/DATASET_GRAMINEA/some_image.jpg

    poetry run python predict.py --model results/dalmax1/.../saved_model.pth \\
        --dir DATA/daninhas_micro/test/DATASET_GRAMINEA --out results/predictions/

Writes, into `--out` (default: alongside the input image, or inside `--dir`
for a folder):

- `predictions.csv`: one row per image, columns `Image Index,Predicted
  Class,Confidence,Path`, plus one probability column per class (header =
  the checkpoint's class names, in order).
- `<stem>.pred.json` per image: full per-class probabilities plus the
  checkpoint's metadata (`model_name`, `class_names`, `img_size`, `extra`).

See `dalmax/inference/predictor.py` for the preprocessing contract (must
match training/testing exactly) and `dalmax.models.checkpoint` for the
checkpoint format, including the clear error raised on a legacy
pre-2026-08-23 checkpoint (see that module's docstring for the historical
save-model bug this fixes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dalmax.inference.export import write_predictions_csv
from dalmax.inference.predictor import PredictionRow, Predictor
from dalmax.models.checkpoint import CheckpointError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dalmax-checkpoint on one image or a folder of images."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to a saved_model.pth checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--dir", type=str, help="Path to a folder of images (searched recursively)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for predictions.csv/*.pred.json (default: alongside the input)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device"
    )
    return parser


def _collect_image_paths(dir_path: Path) -> list[Path]:
    return sorted(
        p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_sidecar_json(row: PredictionRow, predictor: Predictor, out_dir: Path) -> None:
    stem = Path(row.path).stem
    payload = {
        "path": row.path,
        "predicted_class": row.predicted_class,
        "confidence": row.confidence,
        "probabilities": row.probabilities,
        "model": {
            "model_name": predictor.dataset_name,
            "class_names": predictor.class_names,
            "img_size": predictor.img_size,
            "extra": predictor.extra,
        },
    }
    with open(out_dir / f"{stem}.pred.json", "w") as f:
        json.dump(payload, f, indent=2)


def _print_summary(rows: list[PredictionRow]) -> None:
    name_width = max((len(Path(r.path).name) for r in rows), default=10)
    class_width = max((len(r.predicted_class) for r in rows), default=10)
    print(f"{'Image'.ljust(name_width)}  {'Predicted Class'.ljust(class_width)}  Confidence")
    print("-" * (name_width + class_width + 16))
    for row in rows:
        print(
            f"{Path(row.path).name.ljust(name_width)}  "
            f"{row.predicted_class.ljust(class_width)}  {row.confidence:.4f}"
        )
    print(f"\n{len(rows)} image(s) predicted.")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"error: --image path does not exist or is not a file: {image_path}", file=sys.stderr)
            sys.exit(1)
        image_paths = [image_path]
        default_out_dir = image_path.parent
    else:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"error: --dir path does not exist or is not a directory: {dir_path}", file=sys.stderr)
            sys.exit(1)
        image_paths = _collect_image_paths(dir_path)
        default_out_dir = dir_path
        if not image_paths:
            print(f"error: no images found under {dir_path} (extensions: {IMAGE_EXTENSIONS})", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out) if args.out else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        predictor = Predictor(args.model, device=args.device)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = predictor.predict_paths(image_paths)

    csv_path = out_dir / "predictions.csv"
    write_predictions_csv(rows, predictor.class_names, csv_path)
    for row in rows:
        _write_sidecar_json(row, predictor, out_dir)

    _print_summary(rows)
    print(f"\npredictions.csv written to {csv_path}")
    print(f"{len(rows)} *.pred.json sidecar(s) written to {out_dir}")


if __name__ == "__main__":
    main()
