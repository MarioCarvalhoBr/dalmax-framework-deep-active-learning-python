# ADR 0006: A self-describing checkpoint format, plus standalone inference tools

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

`dalmax.models.base.DeepLearning.save_model` did `torch.save(self.net, path)`. `self.net` is the
model *class* passed into `DeepLearning.__init__` (e.g. `DaninhasModelResNet50`), never the
trained instance `self.clf` built inside `.train()`. Every `saved_model.pth` this ever produced —
including every historical run under `results/dalmax1/` and `results/dalmax2/` — is a pickled
reference to the class object only: about 900 bytes, zero weights. `load_model` was also
unimplemented (`# TODO: FIX THIS`): it tried to call the loaded object as `self.net(n_classes)`,
which only accidentally "worked" because `torch.load` on one of these files returns the class
itself, not a real forward pass on trained weights.

This was confirmed directly: re-running `trainer.py` (historical name `demo.py`) on the micro
dataset and inspecting the resulting `saved_model.pth` showed a ~900-byte file that unpickles to
`<class 'dalmax.models.daninhas_resnet50.DaninhasModelResNet50'>`, not a state dict.

Separately, there was no tooling to run a trained model on a new image outside the active-learning
loop (`Strategy.predict` only operates on a `dalmax.data.datasets.Data`-shaped pool), which the
fix above makes newly possible and worth exposing.

## Decision

We will:

1. Introduce `dalmax/models/checkpoint.py` as the single place that knows the on-disk checkpoint
   format (`save_checkpoint`/`load_checkpoint`/`describe_checkpoint`, `CheckpointError`): a plain
   dict — `{"format": "dalmax-checkpoint", "version": 1, "state_dict", "model_name", "n_classes",
   "class_names", "img_size", "extra"}` — saved with `torch.save`. `load_checkpoint` tries
   `torch.load(..., weights_only=True)` first (safe: only a real checkpoint dict unpickles under
   it); on failure it falls back to `weights_only=False` *only* to positively identify the legacy
   bug (a bare `type` object) and raise `CheckpointError` with a message naming the bug and
   instructing re-training, rather than a raw unpickling traceback.
2. Rewire `DeepLearning.save_model`/`load_model` (`dalmax/models/base.py`) and
   `Strategy.save_model` (`dalmax/query_strategies/base.py`) to require `model_name`/`class_names`/
   `img_size`/`extra` as explicit parameters (constructor/call-time injection, never `setattr`) —
   `dalmax.experiment.reporter.write_report` is the only caller and has all four in scope
   (`config.dataset.name`, `result.class_names`, `dalmax.data.registry.get_img_size`, and a
   provenance dict built from `dalmax.experiment.run_metadata.snapshot`).
3. Add `dalmax.models.registry.get_model_class(name)` and `dalmax.data.registry.get_handler(name)`/
   `get_img_size(name)` — thin lookups into the existing `MODEL_REGISTRY`/`DATASET_REGISTRY`/
   `_HANDLER_REGISTRY` dicts, not new if/elif chains, so `load_checkpoint` and the new inference
   tools can rebuild an architecture and its exact preprocessing from a checkpoint's own metadata.
4. Add a new `dalmax/inference/` package (`predictor.py::Predictor`, `export.py`,
   `gui.py::main()`) and three repo-root scripts consuming it: `predict.py` (CLI batch/single-image
   prediction), `loader.py` (checkpoint inspection), `gui.py` (tkinter mini-app). `Predictor`
   replicates training/test preprocessing exactly — resize to the checkpoint's `img_size`, then the
   live dataset handler's own `.transform` (`dalmax.data.registry.get_handler`) — rather than
   re-deriving normalization constants, so it can never silently drift from what a real run used.
5. Rename `demo.py` to `trainer.py` (`git mv`, pure rename, no behavior change) so the training
   entry point and the new inference entry points read as a coherent set at the repo root
   (`trainer.py`, `predict.py`, `loader.py`, `gui.py`).

## Consequences

- Positive: a trained model's weights are finally recoverable and usable outside the training run
  that produced them; `tests/test_checkpoint.py` and `tests/test_golden_run.py`'s new integration
  assertion (checkpoint size + re-prediction matching `predictions.csv`) make regressions here a
  hard test failure, not a silent 900-byte file.
- Positive: checkpoints are self-describing (`model_name`/`class_names`/`img_size`/`extra`), so
  `loader.py`/`predict.py`/`gui.py` need no side-channel config to use one correctly.
- Negative, irreversible: **every checkpoint saved before 2026-08-23 is unrecoverable.** There is
  no weight data hiding anywhere in those files to salvage — the bug never wrote any. This includes
  any `saved_model.pth` under `results/dalmax1/`, `results/dalmax2/`, and any other pre-existing
  results directory. `results.json`/`predictions.csv`/`run_metadata.json`/plots from those runs are
  entirely unaffected (they were never on the buggy code path) — only the weight checkpoint is
  lost. A model must be re-trained (the params JSON + CLI args + seed are recorded in
  `run_metadata.json`, so any historical run is fully reproducible from scratch) if its weights are
  needed. See `.specs/quality/known-issues.md` KI-22.
- Neutral: `dalmax/models/checkpoint.py` introduces a function-local import of
  `dalmax.models.registry` inside `_rebuild_model` (not a top-level import) specifically to avoid a
  circular import (`dalmax.models.registry` imports `dalmax.models.base`, which imports
  `dalmax.models.checkpoint` at module level).
