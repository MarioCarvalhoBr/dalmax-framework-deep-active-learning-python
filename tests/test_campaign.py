"""Tests for `dalmax.campaign` and the committed campaign manifests.

Covers: the committed full/micro manifests load and validate (params load via the
real config loader, no redundant computations, aliases really equal their canonical
run), the manifests match `scripts/campaign/build_manifest.py`, run counts per part,
execution order, the redundancy checker, skip-existing / failure handling (fake
executor, no training), `verify`, and the seed-consistency audit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dalmax import campaign as C
from dalmax.reporting.leaf_check import REQUIRED_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent
FULL = REPO_ROOT / C.DEFAULT_MANIFEST
MICRO = REPO_ROOT / C.DEFAULT_MICRO_MANIFEST


def _builder():
    spec = importlib.util.spec_from_file_location("build_manifest", REPO_ROOT / "scripts/campaign/build_manifest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_manifest"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def full_manifest() -> C.Manifest:
    return C.load_manifest(FULL)


@pytest.fixture(scope="module")
def micro_manifest() -> C.Manifest:
    return C.load_manifest(MICRO)


# --- committed manifests -------------------------------------------------------


def test_committed_manifests_match_the_generator() -> None:
    builder = _builder()
    for rel, scale in builder.TARGETS.items():
        assert (REPO_ROOT / rel).read_text() == builder.render(scale), f"{rel} drifted: rerun the builder"


@pytest.mark.parametrize("which", ["full_manifest", "micro_manifest"])
def test_run_counts_per_part(which: str, request: pytest.FixtureRequest) -> None:
    manifest: C.Manifest = request.getfixturevalue(which)
    counts = C.count_by_part(C.expand_jobs(manifest))
    assert counts == {"paper1": 117, "upper_bound": 3, "rnhal": 42, "texhal": 30}
    assert sum(counts.values()) == 192


def test_paper1_is_12_classical_plus_kmh_and_no_rnhal_texhal_presets(full_manifest: C.Manifest) -> None:
    p1 = [g for g in full_manifest.groups if g.part == "paper1"]
    strategies = {g.strategy_name for g in p1}
    assert "SSRAEKmeansHCSampling" not in strategies and "VCTexKmeansHCSampling" not in strategies
    classical = {g.id for g in p1 if g.strategy_name != "RepresentationStrategy"}
    assert len(classical) == 36  # 35 p1/* groups + shared/random_nq100 (paper-1's RandomSampling@100)
    assert "shared/random_nq100" in classical
    assert {g.id for g in p1 if g.strategy_name == "RepresentationStrategy"} == {
        "p1/KMH/nq10", "p1/KMH/nq50", "shared/kmh_nq100"}


def test_dedup_aliases_point_at_single_canonical_runs(full_manifest: C.Manifest) -> None:
    aliases = full_manifest.alias_by_id()
    for alias in ("texhal/6_3/stage_no_representation", "rnhal/6_3/stage_no_representation", "p1/KMH/nq100"):
        assert aliases[alias].canonical == "shared/kmh_nq100"
    assert aliases["p1/RandomSampling/nq100"].canonical == "shared/random_nq100"
    for row in ("6_1/rep_full", "6_2/ref"):
        assert aliases[f"rnhal/{row}"].canonical == "rnhal/6_3/stage_full"
    for row in ("6_1/rep_full", "6_2/ref", "6_3/stage_full"):
        assert aliases[f"texhal/{row}"].canonical == "shared/texhal_full"
    groups = full_manifest.group_by_id()
    kmh = {(u["paper"], u["table"]) for u in groups["shared/kmh_nq100"].used_by}
    assert {("paper1", "paper1_nq100"), ("paper2", "6_3"), ("paper3", "6_3")} <= kmh
    rnd = {u["paper"] for u in groups["shared/random_nq100"].used_by}
    assert rnd == {"paper1", "paper2", "paper3"}
    tex = {u["paper"] for u in groups["shared/texhal_full"].used_by}
    assert tex == {"paper2", "paper3"}
    # every group consumed by more than one paper lives under shared/
    for gid, g in groups.items():
        if len({u["paper"] for u in g.used_by}) > 1:
            assert gid.startswith("shared/") and "/shared/" in g.dir_results


def test_upper_bound_uses_full_supervised_with_pool_size(full_manifest: C.Manifest, micro_manifest: C.Manifest) -> None:
    ub = full_manifest.group_by_id()["p1/upper_bound"]
    assert (ub.strategy_name, ub.n_init_labeled, ub.n_round) == ("FullSupervised", 8086, 0)
    ub_micro = micro_manifest.group_by_id()["p1/upper_bound"]
    assert (ub_micro.n_init_labeled, ub_micro.n_round) == (806, 0)


def test_no_group_is_a_pure_seed_variant_of_another(full_manifest: C.Manifest) -> None:
    assert C.find_redundant_groups(full_manifest) == []


# --- redundancy checker ----------------------------------------------------------


def test_redundancy_checker_flags_the_same_computation(full_manifest: C.Manifest) -> None:
    canonical = full_manifest.group_by_id()["shared/texhal_full"]
    twin = replace(canonical, id="texhal/6_9/twin", dir_results=canonical.dir_results + "twin/",
                   params_json="files_config/ablations/texhal/rep_full.json", used_by=())
    bad = replace(full_manifest, groups=(*full_manifest.groups, twin))
    assert ("shared/texhal_full", "texhal/6_9/twin") in C.find_redundant_groups(bad)
    with pytest.raises(C.CampaignError, match="redundant"):
        C.validate_manifest(bad)


def test_redundancy_checker_has_no_exceptions(full_manifest: C.Manifest) -> None:
    kmh = full_manifest.group_by_id()["shared/kmh_nq100"]
    twin = replace(kmh, id="rnhal/6_3/stage_no_representation", part="rnhal",
                   dir_results="results/x/", used_by=())
    assert C.find_redundant_groups(replace(full_manifest, groups=(*full_manifest.groups, twin)))


def test_signature_equates_preset_and_generic_equivalent(tmp_path: Path) -> None:
    # The paper-1 micro params differ from the ablation micro ones only by n_drop (smoke speed).
    params = json.loads((REPO_ROOT / "files_config/campaign/params_paper1_micro.json").read_text())
    params["DANINHAS"]["n_drop"] = 10
    same_drop = tmp_path / "params.json"
    same_drop.write_text(json.dumps(params))
    preset = C.effective_signature(str(same_drop), "DANINHAS", "SSRAEKmeansHCSampling", 5, 10, 1)
    generic = C.effective_signature("files_config/ablations/rnhal/micro/stage_full.json", "DANINHAS",
                                    "RepresentationStrategy", 5, 10, 1)
    assert preset == generic


def test_signature_distinguishes_different_computations() -> None:
    args = ("DANINHAS", "RepresentationStrategy", 5, 10, 1)
    a = C.effective_signature("files_config/ablations/rnhal/micro/stage_full.json", *args)
    b = C.effective_signature("files_config/ablations/rnhal/micro/hier_L3.json", *args)
    c = C.effective_signature("files_config/campaign/params_kmh_micro.json", *args)
    assert len({a, b, c}) == 3


def test_alias_that_is_not_the_same_computation_is_rejected(full_manifest: C.Manifest) -> None:
    bad_alias = replace(full_manifest.alias_by_id()["rnhal/6_1/rep_full"],
                        params_json="files_config/ablations/rnhal/hier_L1.json")
    aliases = tuple(bad_alias if a.id == bad_alias.id else a for a in full_manifest.aliases)
    with pytest.raises(C.CampaignError, match="NOT the same computation"):
        C.validate_manifest(replace(full_manifest, aliases=aliases))


def test_unknown_table_run_is_rejected(full_manifest: C.Manifest) -> None:
    tables = json.loads(json.dumps(full_manifest.tables))
    tables["paper1"]["tables"][0]["rows"][0]["run"] = "p1/Nope/nq10"
    with pytest.raises(C.CampaignError, match="unknown run"):
        C.validate_manifest(replace(full_manifest, tables=tables))


# --- expansion / order -----------------------------------------------------------


def test_execution_order_paper1_nq100_first_upper_bound_last(full_manifest: C.Manifest) -> None:
    jobs = C.expand_jobs(full_manifest)
    phases = [C.phase_index(j.group, full_manifest.primary_nq) for j in jobs]
    assert phases == sorted(phases)
    assert jobs[0].group.id == "shared/random_nq100"
    assert all(j.group.n_query == 100 for j in jobs[:39] if j.group.part == "paper1")
    assert jobs[-1].group.part == "upper_bound"
    assert [C.PHASE_LABELS[p] for p in dict.fromkeys(phases)] == [
        "paper1_nq100", "rnhal", "texhal", "paper1_other_nq", "upper_bound"]


def test_parse_parts_and_seed_subset(full_manifest: C.Manifest) -> None:
    assert C.parse_parts("all") == C.PARTS
    assert C.parse_parts("texhal,rnhal") == ("rnhal", "texhal")
    with pytest.raises(C.CampaignError):
        C.parse_parts("paper9")
    jobs = C.expand_jobs(full_manifest, ("rnhal",), seeds=[2])
    assert len(jobs) == 14 and {j.seed for j in jobs} == {2}


# --- running (fake executor) -----------------------------------------------------


def _tiny_manifest(tmp_path: Path) -> C.Manifest:
    def group(gid: str, strategy: str) -> C.RunGroup:
        return C.RunGroup(id=gid, part="paper1", params_json="files_config/campaign/params_paper1_micro.json",
                          strategy_name=strategy, n_query=5, n_init_labeled=10, n_round=1, seeds=(1, 2),
                          dir_results=str(tmp_path / "paper1" / "nq5") + "/", used_by=())
    return C.Manifest(name="tiny", dataset_name="DANINHAS", results_root=str(tmp_path), seeds=(1, 2), primary_nq=5,
                      groups=(group("p1/RandomSampling/nq5", "RandomSampling"),
                              group("p1/MarginSampling/nq5", "MarginSampling")),
                      aliases=(), tables={})


def _write_leaf(leaf: Path, *, omit: tuple[str, ...] = (), initial: str | None = None, n_round: int = 1) -> None:
    leaf.mkdir(parents=True, exist_ok=True)
    payload = {"all_acc": [0.5, 0.6], "all_f1_score": [0.5, 0.6], "all_f1_macro": [0.4, 0.5],
               "rounds": list(range(n_round + 1))}
    files = {
        "results.json": json.dumps(payload),
        "predictions.csv": "Image Index,Real Class,Predicted Class,Correct,Path\n0,a,a,True,x\n",
        "run_metadata.json": json.dumps({"config": {}, "git_commit": "abc"}),
        "log-dalmax.log": f"Initial labeled idxs (sorted): {initial or '[1, 2, 3]'}\n",
    }
    for name in REQUIRED_FILES:
        if name in omit:
            continue
        (leaf / name).write_text(files.get(name, "x"))


def _leaf_of(group: C.RunGroup, seed: int) -> Path:
    return Path(group.dir_results) / "daninhas_micro" / f"SEED_{seed}" / (
        f"NQ_{group.n_query}_NIL_{group.n_init_labeled}_NR_{group.n_round}_NE_1") / group.strategy_name


def test_run_jobs_skip_existing_failures_and_dry_run(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    jobs = C.expand_jobs(manifest)
    assert len(jobs) == 4
    calls: list[list[str]] = []

    def executor(cmd: list[str], env: dict[str, str]) -> int:
        calls.append(cmd)
        assert env["MPLBACKEND"] == "Agg" and env["CUBLAS_WORKSPACE_CONFIG"]
        strategy = cmd[cmd.index("--strategy_name") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        if strategy == "MarginSampling" and seed == 2:
            return 3
        group = next(g for g in manifest.groups if g.strategy_name == strategy)
        _write_leaf(_leaf_of(group, seed))
        return 0

    dry = C.run_jobs(jobs, manifest, device="cpu", dry_run=True, executor=executor, echo=lambda _: None)
    assert (dry.dry, calls) == (4, [])

    first = C.run_jobs(jobs, manifest, device="cpu", executor=executor, echo=lambda _: None)
    assert (first.executed, first.failed, first.skipped) == (3, 1, 0)
    failures = (tmp_path / "failures.log").read_text()
    assert "p1/MarginSampling/nq5 seed=2" in failures
    assert "FAILED" in (tmp_path / "campaign.log").read_text()

    calls.clear()
    second = C.run_jobs(jobs, manifest, device="cpu", executor=executor, echo=lambda _: None)
    assert (second.skipped, second.executed, second.failed) == (3, 0, 1)  # only the failed job is retried
    assert len(calls) == 1


def test_build_command_matches_the_ablation_scripts_conventions(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    cmd = C.build_command(C.Job(manifest.groups[0], 2), manifest, "cuda", python="python")
    assert cmd[:2] == ["python", "tools/trainer.py"]
    assert cmd[cmd.index("--seed") + 1] == "2" and cmd[cmd.index("--device") + 1] == "cuda"
    assert cmd[cmd.index("--n_round") + 1] == "1" and cmd[cmd.index("--n_init_labeled") + 1] == "10"


def test_env_pins_cuda_visible_devices_default_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert C.build_env("cuda")["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert C.build_env("cuda")["CUDA_VISIBLE_DEVICES"] == "1"


# --- verify ----------------------------------------------------------------------


def test_verify_job_ok_incomplete_missing(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    g = manifest.groups[0]
    _write_leaf(_leaf_of(g, 1))
    _write_leaf(_leaf_of(g, 2), omit=("predictions.csv",))
    statuses = {j.seed: C.verify_job(j).status for j in C.expand_jobs(manifest, seeds=[1, 2]) if j.group is g}
    assert statuses == {1: "OK", 2: "INCOMPLETE"}
    other = next(j for j in C.expand_jobs(manifest) if j.group.strategy_name == "MarginSampling")
    assert C.verify_job(other).status == "MISSING"
    assert "predictions.csv" in C.render_verify([C.verify_job(j) for j in C.expand_jobs(manifest)])


# --- seed-consistency audit ------------------------------------------------------


def test_seed_audit_pass_fail_and_full_supervised_excluded(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    ub = replace(manifest.groups[0], id="p1/upper_bound", part="upper_bound", strategy_name="FullSupervised",
                 n_init_labeled=99, n_round=0, dir_results=str(tmp_path / "ub") + "/")
    manifest = replace(manifest, groups=(*manifest.groups, ub))
    g_random, g_margin = manifest.groups[0], manifest.groups[1]
    for seed in (1, 2):
        _write_leaf(_leaf_of(g_random, seed), initial="[1, 2, 3]")
        _write_leaf(_leaf_of(ub, seed), initial="[1, 2, 3, 4, 5]", n_round=0)  # whole pool: excluded
    _write_leaf(_leaf_of(g_margin, 1), initial="[1, 2, 3]")
    _write_leaf(_leaf_of(g_margin, 2), initial="[9, 9, 9]")  # seed 2: offender

    audits = {a.seed: a for a in C.audit_seed_consistency(C.expand_jobs(manifest))}
    assert audits[1].status == "PASS" and audits[1].n_runs == 2
    assert audits[2].status == "FAIL" and audits[2].offenders == ("p1/MarginSampling/nq5",)
    text = C.render_seed_audit(list(audits.values()))
    assert "seed 2: FAIL" in text and "OFFENDER" in text and "p1/MarginSampling/nq5" in text


def test_seed_audit_no_data_when_nothing_ran(tmp_path: Path) -> None:
    audits = C.audit_seed_consistency(C.expand_jobs(_tiny_manifest(tmp_path)))
    assert {a.status for a in audits} == {"NO_DATA"}


def test_read_initial_idxs(tmp_path: Path) -> None:
    (tmp_path / "log-dalmax.log").write_text("noise\n2026 - WARNING - Initial labeled idxs (sorted): [4, 8, 15]\n")
    assert C.read_initial_idxs(tmp_path) == [4, 8, 15]
    assert C.read_initial_idxs(tmp_path / "nope") is None


def test_cli_list_and_parts(capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main(["list", "--micro", "--seeds", "1", "--part", "rnhal,texhal"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "24 runs" in out
    assert C.main(["list", "--part", "nope"]) == 2


# --- resume: only a COMPLETE leaf counts as done (A2) ------------------------------


def test_job_done_requires_a_complete_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _tiny_manifest(tmp_path)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)  # find_leaf resolves dir_results against it
    job = C.Job(manifest.groups[0], 1)
    assert not C.job_done(job)  # no leaf at all
    _write_leaf(_leaf_of(job.group, 1), omit=("saved_model.pth",))
    assert (_leaf_of(job.group, 1) / "results.json").is_file()
    assert not C.job_done(job)  # results.json exists but the run did not finish writing
    _write_leaf(_leaf_of(job.group, 1))
    assert C.job_done(job)


def test_reporter_writes_results_json_last() -> None:
    import ast

    tree = ast.parse((REPO_ROOT / "dalmax/experiment/reporter.py").read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "write_report")
    calls = [n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index("_write_results_json") > calls.index("_write_predictions_csv")
    save = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Attribute) and n.attr == "save_model"]
    results = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == "_write_results_json"]
    assert save and results and max(save) < min(results)


# --- seed audit WARN + verify exit codes (A3) ------------------------------------------


def test_seed_audit_warns_on_missing_log_and_exit_codes(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    g_random, g_margin = manifest.groups
    for seed in (1, 2):
        _write_leaf(_leaf_of(g_random, seed), initial="[1, 2, 3]")
        _write_leaf(_leaf_of(g_margin, seed), initial="[1, 2, 3]")
    jobs = C.expand_jobs(manifest)
    ok = [C.verify_job(j) for j in jobs]
    audits = C.audit_seed_consistency(jobs)
    assert {a.status for a in audits} == {"PASS"} and C.verify_exit_code(ok, audits) == 0

    (_leaf_of(g_margin, 2) / "log-dalmax.log").unlink()  # missing log: WARN, not silence
    # `log-dalmax.log` is a required artifact, so make the leaf otherwise complete again
    (_leaf_of(g_margin, 2) / "log-dalmax.log").write_text("no initial idxs line here\n")
    jobs = C.expand_jobs(manifest)
    audits = {a.seed: a for a in C.audit_seed_consistency(jobs)}
    assert audits[1].status == "PASS" and audits[2].status == "WARN"
    assert audits[2].unreadable == ("p1/MarginSampling/nq5",)
    statuses = [C.verify_job(j) for j in jobs]
    assert C.verify_exit_code(statuses, list(audits.values())) == 2

    _write_leaf(_leaf_of(g_margin, 2), initial="[7, 7]")  # a real mismatch: FAIL
    audits = C.audit_seed_consistency(C.expand_jobs(manifest))
    assert {a.seed: a.status for a in audits}[2] == "FAIL"
    assert C.verify_exit_code([C.verify_job(j) for j in C.expand_jobs(manifest)], audits) == 1


def test_seed_audit_warn_when_no_log_is_readable(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    g = manifest.groups[0]
    _write_leaf(_leaf_of(g, 1))
    (_leaf_of(g, 1) / "log-dalmax.log").unlink()
    audits = {a.seed: a for a in C.audit_seed_consistency(C.expand_jobs(manifest))}
    assert audits[1].status == "WARN" and audits[2].status == "NO_DATA"


# --- --exclude-strategy (CPU smoke skips the two adversarial baselines) ------------------------


ADVERSARIAL = ("AdversarialBIM", "AdversarialDeepFool")


def test_exclude_strategy_filter_drops_only_those_groups(full_manifest: C.Manifest) -> None:
    jobs = C.expand_jobs(full_manifest)
    kept = C.expand_jobs(full_manifest, exclude_strategies=ADVERSARIAL)
    dropped = [j for j in jobs if j not in kept]
    assert dropped and {j.group.strategy_name for j in dropped} == set(ADVERSARIAL)
    assert len(dropped) == 2 * 3 * 3  # 2 strategies x 3 n_query x 3 seeds
    assert len(kept) == len(jobs) - len(dropped) == 192 - 18
    assert all(j.group.strategy_name not in ADVERSARIAL for j in kept)
    assert [j for j in jobs if j in kept] == kept  # order preserved


def test_exclude_strategy_parsing_fails_fast_on_unknown_names(full_manifest: C.Manifest) -> None:
    assert C.parse_exclude_strategies(None, full_manifest) == frozenset()
    assert C.parse_exclude_strategies("AdversarialBIM, AdversarialDeepFool", full_manifest) == frozenset(ADVERSARIAL)
    with pytest.raises(C.CampaignError, match="unknown strategy"):
        C.parse_exclude_strategies("NoSuchStrategy", full_manifest)


def test_cli_exclude_strategy_applies_to_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main(["list", "--micro", "--seeds", "1", "--exclude-strategy", ",".join(ADVERSARIAL)]) == 0
    out = capsys.readouterr().out
    assert "excluded strategies: AdversarialBIM, AdversarialDeepFool" in out
    assert "Adversarial" not in out.split("excluded strategies:")[1].split("\n", 1)[1]
    assert "TOTAL" in out and "58 runs" in out  # 64 micro seed-1 jobs - 2 strategies x 3 n_query
    assert C.main(["list", "--micro", "--exclude-strategy", "Nope"]) == 2


def test_verify_with_exclusion_does_not_report_excluded_groups_missing(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    for seed in (1, 2):
        _write_leaf(_leaf_of(manifest.groups[0], seed))
    excluded = C.expand_jobs(manifest, exclude_strategies=("MarginSampling",))
    statuses = [C.verify_job(j) for j in excluded]
    assert all(s.status == "OK" for s in statuses) and len(statuses) == 2
    assert C.verify_exit_code(statuses, C.audit_seed_consistency(excluded)) == 0


def test_cli_broken_pipe_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_args) -> int:
        raise BrokenPipeError

    monkeypatch.setattr(C, "_dispatch", boom)
    monkeypatch.setattr(C.os, "dup2", lambda *a, **k: None)
    monkeypatch.setattr(C.sys, "stdout", type("S", (), {"fileno": lambda self: 1})())
    assert C.main(["list"]) == 0
