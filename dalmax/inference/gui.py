"""Tkinter mini-app for running a `dalmax-checkpoint` on images interactively.

All `tkinter` construction happens inside `main()` — importing this module
must stay side-effect-free (no window, no display connection required) so it
is safe to `import` in headless CI (`tests/test_gui_helpers.py` does exactly
that). The root shim is `gui.py` at the repo root.

Workflow
--------
1. "Select model (.pth)" loads a checkpoint via `dalmax.inference.predictor.Predictor`
   and shows its class names / dataset / strategy provenance.
2. "Select image(s)" / "Select folder" pick input images.
3. "Run prediction" fills a results table (file, predicted class, confidence%).
4. Selecting a row previews that image and its per-class probabilities.
5. "Export CSV" writes the same schema as `predict.py`
   (`dalmax.inference.export.write_predictions_csv`).

Errors (in particular a legacy pre-2026-08-23 checkpoint, see
`dalmax.models.checkpoint`'s module docstring) are shown via
`tkinter.messagebox`, never raised past the button handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dalmax.inference.export import write_predictions_csv
from dalmax.inference.predictor import PredictionRow, Predictor
from dalmax.models.checkpoint import CheckpointError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
PREVIEW_MAX_SIDE = 300


def collect_image_paths(dir_path: str | Path) -> list[Path]:
    """Recursively find every image file under `dir_path`, sorted for a
    stable, reproducible display/prediction order. Pure logic, no `tkinter`
    dependency, so it is unit-testable headlessly."""
    root = Path(dir_path)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def format_model_summary(predictor: Predictor) -> str:
    """Build the label text shown after loading a model: classes, dataset,
    strategy. Pure logic, no `tkinter` dependency."""
    extra = predictor.extra
    strategy = extra.get("strategy_name", "unknown")
    dataset = extra.get("dataset_name", predictor.dataset_name)
    seed = extra.get("seed", "unknown")
    lines = [
        f"Dataset: {dataset}   Strategy: {strategy}   Seed: {seed}",
        f"Classes ({len(predictor.class_names)}): {', '.join(predictor.class_names)}",
        f"Image size: {predictor.img_size}x{predictor.img_size}",
    ]
    return "\n".join(lines)


def format_confidence_percent(confidence: float) -> str:
    """`0.8734` -> `"87.34%"`. Pure logic, no `tkinter` dependency."""
    return f"{confidence * 100:.2f}%"


def format_probabilities(row: PredictionRow) -> str:
    """Multi-line `"class_name: 87.34%"` block, sorted by probability
    descending, for the per-class-probabilities panel."""
    ordered = sorted(row.probabilities.items(), key=lambda kv: -kv[1])
    return "\n".join(f"{name}: {format_confidence_percent(prob)}" for name, prob in ordered)


class _GuiState:
    """Non-`tkinter` state the button handlers close over: the loaded
    `Predictor` (if any), the current image paths, and the last prediction
    rows (for the row-selection preview and CSV export)."""

    def __init__(self) -> None:
        self.predictor: Predictor | None = None
        self.image_paths: list[Path] = []
        self.rows: list[PredictionRow] = []


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        from PIL import Image, ImageTk
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Pillow (PIL) is required for the DalMax inference GUI") from exc

    state = _GuiState()

    root = tk.Tk()
    root.title("DalMax — Model Inference")
    root.geometry("980x640")

    status_var = tk.StringVar(value="Load a model checkpoint to begin.")
    model_summary_var = tk.StringVar(value="No model loaded.")
    probabilities_var = tk.StringVar(value="")

    top_frame = ttk.Frame(root, padding=8)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    middle_frame = ttk.Frame(root, padding=8)
    middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(text: str) -> None:
        status_var.set(text)
        root.update_idletasks()

    def on_select_model() -> None:
        path = filedialog.askopenfilename(
            title="Select model checkpoint", filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*")]
        )
        if not path:
            return
        try:
            predictor = Predictor(path, device="auto")
        except CheckpointError as exc:
            messagebox.showerror("Failed to load checkpoint", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user, not a crash
            messagebox.showerror("Failed to load checkpoint", str(exc))
            return
        state.predictor = predictor
        model_summary_var.set(format_model_summary(predictor))
        set_status(f"Model loaded: {path}")

    def on_select_images() -> None:
        paths = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[("Images", " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)), ("All files", "*")],
        )
        if not paths:
            return
        state.image_paths = [Path(p) for p in paths]
        set_status(f"{len(state.image_paths)} image(s) selected.")

    def on_select_folder() -> None:
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        state.image_paths = collect_image_paths(folder)
        if not state.image_paths:
            messagebox.showwarning("No images found", f"No supported images found under {folder}")
            return
        set_status(f"{len(state.image_paths)} image(s) found under {folder}.")

    def on_run_prediction() -> None:
        if state.predictor is None:
            messagebox.showerror("No model loaded", "Select a model checkpoint first.")
            return
        if not state.image_paths:
            messagebox.showerror("No images selected", "Select image(s) or a folder first.")
            return
        set_status(f"Predicting on {len(state.image_paths)} image(s)...")
        try:
            state.rows = state.predictor.predict_paths(state.image_paths)
        except Exception as exc:  # noqa: BLE001 - surface any inference error to the user
            messagebox.showerror("Prediction failed", str(exc))
            return

        for item in results_tree.get_children():
            results_tree.delete(item)
        for row in state.rows:
            results_tree.insert(
                "",
                tk.END,
                values=(Path(row.path).name, row.predicted_class, format_confidence_percent(row.confidence)),
            )
        probabilities_var.set("")
        preview_label.configure(image="", text="(select a row to preview)")
        set_status(f"Predicted {len(state.rows)} image(s).")

    def on_row_selected(_event: Any) -> None:
        selection = results_tree.selection()
        if not selection:
            return
        index = results_tree.index(selection[0])
        row = state.rows[index]

        probabilities_var.set(format_probabilities(row))

        try:
            image = Image.open(row.path).convert("RGB")
            image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
            photo = ImageTk.PhotoImage(image)
        except Exception as exc:  # noqa: BLE001 - a bad image file must not crash the GUI
            messagebox.showerror("Failed to preview image", str(exc))
            return
        preview_label.configure(image=photo, text="")
        preview_label.image = photo  # keep a reference alive

    def on_export_csv() -> None:
        if not state.rows or state.predictor is None:
            messagebox.showerror("Nothing to export", "Run a prediction first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Export predictions CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="predictions.csv",
        )
        if not out_path:
            return
        write_predictions_csv(state.rows, state.predictor.class_names, out_path)
        set_status(f"Predictions exported to {out_path}")

    ttk.Button(top_frame, text="Select model (.pth)", command=on_select_model).pack(side=tk.LEFT, padx=4)
    ttk.Button(top_frame, text="Select image(s)", command=on_select_images).pack(side=tk.LEFT, padx=4)
    ttk.Button(top_frame, text="Select folder", command=on_select_folder).pack(side=tk.LEFT, padx=4)
    ttk.Button(top_frame, text="Run prediction", command=on_run_prediction).pack(side=tk.LEFT, padx=4)
    ttk.Button(top_frame, text="Export CSV", command=on_export_csv).pack(side=tk.LEFT, padx=4)

    model_summary_label = ttk.Label(root, textvariable=model_summary_var, padding=(8, 0), justify=tk.LEFT)
    model_summary_label.pack(side=tk.TOP, fill=tk.X, before=middle_frame)

    results_frame = ttk.Frame(middle_frame)
    results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    columns = ("file", "predicted_class", "confidence")
    results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
    results_tree.heading("file", text="File")
    results_tree.heading("predicted_class", text="Predicted Class")
    results_tree.heading("confidence", text="Confidence")
    results_tree.bind("<<TreeviewSelect>>", on_row_selected)
    results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_tree.yview)
    results_tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    detail_frame = ttk.Frame(middle_frame, padding=(8, 0))
    detail_frame.pack(side=tk.LEFT, fill=tk.Y)

    preview_label = ttk.Label(detail_frame, text="(select a row to preview)", anchor=tk.CENTER)
    preview_label.pack(side=tk.TOP, pady=(0, 8))

    ttk.Label(detail_frame, text="Per-class probabilities:").pack(side=tk.TOP, anchor=tk.W)
    ttk.Label(detail_frame, textvariable=probabilities_var, justify=tk.LEFT).pack(side=tk.TOP, anchor=tk.W)

    root.mainloop()


if __name__ == "__main__":
    main()
