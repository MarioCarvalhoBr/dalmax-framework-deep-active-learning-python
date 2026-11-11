"""Superseded leaves of the re-run parts (ADR 0014): `dalmax.superseded.classify_leaf`, the
`dalmax.campaign` run guard and the `STALE` verify status. Temporary trees only, never the real
`results/`."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dalmax import campaign as C
from dalmax import superseded as S
from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.reporting.leaf_check import REQUIRED_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent
MICRO_PARAMS = "files_config/campaign/params_paper1_micro.json"
OLD = "2026-10-01T10:00:00.000000+00:00"
NEW = "2026-11-12T08:00:00.000000+00:00"


def _metadata(started_at: object, *, fixed: bool) -> dict:
    dataset = {"name": "DANINHAS", **({"model_args": {"dropout_p": None}} if fixed else {})}
    return {"started_at": started_at, "config": {"dataset": dataset}, "git_commit": "abc"}


def _write_leaf(leaf: Path, metadata: dict | str | None, *, files: tuple[str, ...] = REQUIRED_FILES) -> Path:
    leaf.mkdir(parents=True, exist_ok=True)
    for name in files:
        if name == "run_metadata.json":
            continue
        content = "Initial labeled idxs (sorted): [1, 2, 3]\n" if name == "log-dalmax.log" else "x"
        if name == "results.json":
            content = json.dumps({"all_acc": [0.5, 0.6], "all_f1_score": [0.5, 0.6],
                                  "all_f1_macro": [0.4, 0.5], "rounds": [0, 1]})
        if name == "predictions.csv":
            content = "Image Index,Real Class,Predicted Class,Correct,Path\n0,a,a,True,x\n"
        (leaf / name).write_text(content)
    if metadata is not None:
        text = metadata if isinstance(metadata, str) else json.dumps(metadata)
        (leaf / "run_metadata.json").write_text(text)
    return leaf


def _manifest(tmp_path: Path) -> C.Manifest:
    def group(gid: str, part: str, strategy: str) -> C.RunGroup:
        return C.RunGroup(id=gid, part=part, params_json=MICRO_PARAMS, strategy_name=strategy, n_query=5,
                          n_init_labeled=10, n_round=1, seeds=(1, 2),
                          dir_results=str(tmp_path / "paper1" / "nq5") + "/", used_by=())
    return C.Manifest(name="tiny", dataset_name="DANINHAS", results_root=str(tmp_path), seeds=(1, 2), primary_nq=5,
                      groups=(group("p1/KMeansSampling/nq5", "paper1_rerun_a", "KMeansSampling"),
                              group("p1/BALDDropout/nq5", "paper1_rerun_b", "BALDDropout"),
                              group("p1/RandomSampling/nq5", "paper1", "RandomSampling")),
                      aliases=(), tables={})


def _leaf_of(group: C.RunGroup, seed: int) -> Path:
    return Path(group.dir_results) / "daninhas_micro" / f"SEED_{seed}" / (
        f"NQ_{group.n_query}_NIL_{group.n_init_labeled}_NR_{group.n_round}_NE_1") / group.strategy_name


def _by_strategy(manifest: C.Manifest, strategy: str) -> C.RunGroup:
    return next(g for g in manifest.groups if g.strategy_name == strategy)


# --- one definition -------------------------------------------------------------------


def test_engine_and_builder_take_the_part_names_from_the_shared_module() -> None:
    assert set(S.RERUN_PARTS) == {"paper1_rerun_a", "paper1_rerun_b"}
    assert all(p in C.PARTS and p in C.PHASE_LABELS for p in S.RERUN_PARTS)
    assert C.RERUN_PARTS is S.RERUN_PARTS
    assert S.default_cutoff() == datetime(2026, 11, 11, 19, 12, 28, tzinfo=timezone.utc)


def test_fixed_code_always_writes_the_marker_key() -> None:
    # Claim (b) of the definition: the resolved config of ANY params JSON carries
    # config.dataset.model_args (null when the block is absent), so a fixed-code leaf has it.
    for params in (MICRO_PARAMS, "files_config/campaign/params_paper1.json"):
        config = load_experiment_config(REPO_ROOT / params, "DANINHAS", strategy_name="RandomSampling", seed=1,
                                        n_init_labeled=10, n_query=5, n_round=1, dir_results="x/", device="cpu")
        assert "model_args" in to_dict(config)["dataset"]


# --- classify_leaf --------------------------------------------------------------------


def test_classify_current_and_both_superseded_rules(tmp_path: Path) -> None:
    current = S.classify_leaf(_write_leaf(tmp_path / "a", _metadata(NEW, fixed=True)))
    assert current.status == S.CURRENT and not current.blocking
    before = S.classify_leaf(_write_leaf(tmp_path / "b", _metadata(OLD, fixed=True)))
    assert before.status == S.SUPERSEDED and "before the fix cutoff" in before.reason and before.blocking
    unmarked = S.classify_leaf(_write_leaf(tmp_path / "c", _metadata(NEW, fixed=False)))
    assert unmarked.status == S.SUPERSEDED and "lacks config.dataset.model_args" in unmarked.reason
    # An explicit cutoff moves rule (a); rule (b) still holds.
    late = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert S.classify_leaf(tmp_path / "a", late).status == S.SUPERSEDED


@pytest.mark.parametrize("metadata", [None, "{not json", "[1, 2]", _metadata("2026-11-12T08:00:00", fixed=True)])
def test_classify_without_readable_metadata(tmp_path: Path, metadata: object) -> None:
    with_results = S.classify_leaf(_write_leaf(tmp_path / "done", metadata))
    assert with_results.status == S.UNREADABLE and with_results.blocking
    interrupted = S.classify_leaf(_write_leaf(tmp_path / "cut", metadata, files=("log-dalmax.log",)))
    assert interrupted.status == S.INTERRUPTED and not interrupted.blocking


# --- run guard ------------------------------------------------------------------------


def _executor(calls: list[list[str]]):
    def run(cmd: list[str], env: dict[str, str]) -> int:
        calls.append(cmd)
        return 0
    return run


@pytest.mark.parametrize("kwargs", [{}, {"dry_run": True}, {"skip_existing": False}])
def test_run_refuses_before_launching_anything(tmp_path: Path, kwargs: dict) -> None:
    manifest = _manifest(tmp_path)
    _write_leaf(_leaf_of(_by_strategy(manifest, "KMeansSampling"), 1), _metadata(OLD, fixed=False))
    calls: list[list[str]] = []
    with pytest.raises(C.SupersededLeavesError, match="refusing to run: 1 leaf") as err:
        C.run_jobs(C.expand_jobs(manifest), manifest, device="cpu", executor=_executor(calls),
                   echo=lambda _: None, **kwargs)
    assert "make campaign-archive-superseded" in str(err.value) and "SUPERSEDED" in str(err.value)
    assert calls == [] and not (tmp_path / "campaign.log").exists()


def test_run_guard_lists_the_count_and_the_first_leaves(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    for strategy in ("KMeansSampling", "BALDDropout"):
        for seed in (1, 2):
            _write_leaf(_leaf_of(_by_strategy(manifest, strategy), seed), _metadata(OLD, fixed=False))
    with pytest.raises(C.SupersededLeavesError) as err:
        C.ensure_no_blocking_leaves(C.expand_jobs(manifest), show=3)
    text = str(err.value)
    assert "refusing to run: 4 leaf" in text and "... and 1 more" in text
    assert text.count("SUPERSEDED ") == 3


def test_run_guard_ignores_other_parts_and_passes_current_and_interrupted_leaves(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    random_group = _by_strategy(manifest, "RandomSampling")
    kmeans, bald = _by_strategy(manifest, "KMeansSampling"), _by_strategy(manifest, "BALDDropout")
    for seed in (1, 2):
        _write_leaf(_leaf_of(random_group, seed), _metadata(OLD, fixed=False))  # part paper1: never inspected
        _write_leaf(_leaf_of(kmeans, seed), _metadata(NEW, fixed=True))  # current, complete: skipped
    _write_leaf(_leaf_of(bald, 1), None, files=("log-dalmax.log",))  # interrupted: re-run
    calls: list[list[str]] = []
    summary = C.run_jobs(C.expand_jobs(manifest), manifest, device="cpu", executor=_executor(calls),
                         echo=lambda _: None)
    assert (summary.skipped, summary.executed) == (4, 2)
    assert sorted(c[c.index("--seed") + 1] for c in calls) == ["1", "2"]
    assert all(c[c.index("--strategy_name") + 1] == "BALDDropout" for c in calls)


def test_run_guard_checks_every_matching_leaf(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    kmeans = _by_strategy(manifest, "KMeansSampling")
    _write_leaf(_leaf_of(kmeans, 1), _metadata(NEW, fixed=True))
    other = Path(str(_leaf_of(kmeans, 1)).replace("daninhas_micro", "zz_other_folder"))
    _write_leaf(other, _metadata(OLD, fixed=False))  # not the first match of find_leaf
    found = C.blocking_leaves(C.expand_jobs(manifest))
    assert [b.leaf for b in found] == [other]


# --- verify ---------------------------------------------------------------------------


def test_verify_reports_stale_and_fails(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    kmeans, bald = _by_strategy(manifest, "KMeansSampling"), _by_strategy(manifest, "BALDDropout")
    random_group = _by_strategy(manifest, "RandomSampling")
    _write_leaf(_leaf_of(kmeans, 1), _metadata(OLD, fixed=False))  # superseded
    _write_leaf(_leaf_of(kmeans, 2), _metadata(NEW, fixed=True))  # current
    _write_leaf(_leaf_of(bald, 1), None)  # results.json without metadata: unreadable
    _write_leaf(_leaf_of(bald, 2), None, files=("log-dalmax.log",))  # interrupted
    for seed in (1, 2):
        _write_leaf(_leaf_of(random_group, seed), _metadata(OLD, fixed=False))  # other part: OK
    statuses = {(s.job.group.strategy_name, s.job.seed): s for s in map(C.verify_job, C.expand_jobs(manifest))}
    assert {k: s.status for k, s in statuses.items()} == {
        ("KMeansSampling", 1): "STALE", ("KMeansSampling", 2): "OK",
        ("BALDDropout", 1): "STALE", ("BALDDropout", 2): "INCOMPLETE",
        ("RandomSampling", 1): "OK", ("RandomSampling", 2): "OK",
    }
    assert "unreadable" in statuses[("BALDDropout", 1)].detail
    text = C.render_verify(list(statuses.values()))
    assert "Totals: OK=3 INCOMPLETE=1 MISSING=0 STALE=2 (of 6 jobs)" in text
    assert "make campaign-archive-superseded" in text
    only_stale = [statuses[("KMeansSampling", 1)]]
    assert C.verify_exit_code(only_stale, [C.SeedAudit(1, "PASS", 1)]) == 1


# --- CLI ------------------------------------------------------------------------------


def _write_manifest(tmp_path: Path) -> Path:
    manifest = _manifest(tmp_path)
    raw = {
        "name": "tiny", "dataset_name": "DANINHAS", "results_root": str(tmp_path), "seeds": [1, 2],
        "primary_nq": 5, "aliases": [], "tables": {},
        "groups": [{"id": g.id, "part": g.part, "params_json": g.params_json, "strategy_name": g.strategy_name,
                    "n_query": g.n_query, "n_init_labeled": g.n_init_labeled, "n_round": g.n_round,
                    "dir_results": g.dir_results, "used_by": []} for g in manifest.groups],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    return path


def test_cli_run_dry_run_refuses_with_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_manifest(tmp_path)
    manifest = _manifest(tmp_path)
    _write_leaf(_leaf_of(_by_strategy(manifest, "BALDDropout"), 2), _metadata(OLD, fixed=False))
    assert C.main(["run", "--manifest", str(path), "--part", "paper1_rerun_b", "--dry-run", "--device", "cpu"]) == 3
    captured = capsys.readouterr()
    assert "refusing to run: 1 leaf" in captured.err and "DRY-RUN" not in captured.out
    # Another part is not affected by the stale leaf.
    assert C.main(["run", "--manifest", str(path), "--part", "paper1", "--dry-run", "--device", "cpu"]) == 0
    assert C.main(["verify", "--manifest", str(path), "--part", "paper1_rerun_b"]) == 1
    assert "STALE      p1/BALDDropout/nq5 seed=2" in capsys.readouterr().out


def test_tiny_manifest_is_valid(tmp_path: Path) -> None:
    loaded = C.load_manifest(_write_manifest(tmp_path))
    assert replace(loaded, path="") == _manifest(tmp_path)
