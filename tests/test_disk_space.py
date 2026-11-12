"""`dalmax.experiment.disk_space` and the trainer's no-space exit code (2026-11-12, Drive quota gate)."""

from __future__ import annotations

import errno
import types
from pathlib import Path

import pytest

from dalmax.experiment import disk_space as D


def test_required_bytes_is_runs_times_size_times_margin() -> None:
    assert D.required_bytes(0) == 0
    assert D.required_bytes(10, run_bytes=100, margin=1.2) == 1200
    assert D.required_bytes(282) == int(282 * D.RUN_DISK_BYTES * D.DISK_MARGIN)


def test_free_bytes_of_a_root_that_does_not_exist_yet(tmp_path: Path) -> None:
    missing = tmp_path / "results" / "campaign_fixed"
    assert D.existing_ancestor(missing) == tmp_path
    assert D.free_bytes(missing) > 0
    assert D.file_system_id(missing) == D.file_system_id(tmp_path)


def test_is_no_space_error_on_errno_message_and_chain() -> None:
    assert D.is_no_space_error(OSError(errno.ENOSPC, "No space left on device"))
    assert D.is_no_space_error(OSError(errno.EDQUOT, "Disk quota exceeded"))
    assert D.is_no_space_error(RuntimeError("PytorchStreamWriter failed writing file: No space left on device"))
    try:
        try:
            raise OSError(errno.ENOSPC, "full")
        except OSError as inner:
            raise RuntimeError("file write failed") from inner
    except RuntimeError as wrapped:
        assert D.is_no_space_error(wrapped)
    assert not D.is_no_space_error(ValueError("bad config"))
    assert not D.is_no_space_error(OSError(errno.ENOENT, "missing"))


def test_trainer_cli_exits_with_the_no_space_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import dalmax.cli as cli

    class _Logger:
        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            self.last_error = msg

    logger = _Logger()

    class _Runner:
        def __init__(self, config, logger) -> None:
            pass

        def run(self):
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cli, "get_logger", lambda: logger)
    monkeypatch.setattr(cli, "get_path_logger", lambda: "unused.log")
    monkeypatch.setattr(cli, "load_experiment_config", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(cli, "to_dict", lambda config: {})
    monkeypatch.setattr(cli, "_apply_embedding_variant_override", lambda config, variant: config)
    monkeypatch.setattr(cli, "ExperimentRunner", _Runner)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--dataset_name", "DANINHAS"])
    assert exc.value.code == D.NO_SPACE_EXIT_CODE == 28
    assert "No space left" in logger.last_error

    class _OtherFailure(_Runner):
        def run(self):
            raise ValueError("unrelated")

    monkeypatch.setattr(cli, "ExperimentRunner", _OtherFailure)
    with pytest.raises(ValueError, match="unrelated"):  # any other failure propagates unchanged
        cli.main(["--dataset_name", "DANINHAS"])
