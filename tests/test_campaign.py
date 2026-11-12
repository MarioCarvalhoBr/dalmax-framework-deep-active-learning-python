"""Tests for `dalmax.campaign` and the committed campaign manifests.

Covers: the committed full/micro manifests load and validate (params load via the
real config loader, no redundant computations, aliases really equal their canonical
run), the manifests match `scripts/campaign/build_manifest.py`, run counts per part,
execution order, the GPU parts and their estimated balance (ADR 0016), the
redundancy checker and the signature, skip-existing / failure handling (fake
executor, no training), `verify` (including the manifest's required per-round
files), and the seed-consistency audit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dalmax import campaign as C
from dalmax.reporting.leaf_check import REQUIRED_FILES, ROUND_HISTORY_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent
FULL = REPO_ROOT / C.DEFAULT_MANIFEST
MICRO = REPO_ROOT / C.DEFAULT_MICRO_MANIFEST
SWEEP = REPO_ROOT / "files_config/campaign/manifest_kmh_sweep.json"
SWEEP_MICRO = REPO_ROOT / "files_config/campaign/manifest_kmh_sweep_micro.json"
MC_DROPOUT = ("LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout", "BALDDropout")
DROPOUT_FAMILY = (*MC_DROPOUT, "VarRatio", "MeanSTD")
DEEPALPLUS = ("VarRatio", "MeanSTD", "KCenterGreedyPCA", "BadgeSampling", "CEALSampling", "LossPredictionLoss",
              "VAAL", "WAAL")
LONG = ("AdversarialBIM", "VAAL", "WAAL", "AdversarialDeepFool")


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
    for rel, target in builder.TARGETS.items():
        assert (REPO_ROOT / rel).read_text() == builder.render(target), f"{rel} drifted: rerun the builder"


def test_control_ki43_manifest_expands_to_three_jobs() -> None:
    manifest = C.load_manifest(REPO_ROOT / "files_config/campaign/manifest_control_ki43.json")
    jobs = C.expand_jobs(manifest, C.parse_parts("control"))
    assert [(j.group.id, j.seed) for j in jobs] == [("control/random_nq100_ki43", s) for s in (1, 2, 3)]
    assert manifest.results_root == "results/campaign_control_fixed"
    assert jobs[0].group.dir_results == "results/campaign_control_fixed/random_nq100/"


@pytest.mark.parametrize("path", [SWEEP, SWEEP_MICRO])
def test_kmh_sweep_manifest_counts_and_layout(path: Path) -> None:
    manifest = C.load_manifest(path)  # validates: params load, no redundant computation, aliases == canonical
    counts = C.count_by_part(C.expand_jobs(manifest))
    assert counts["kmh_nq10"] == counts["kmh_nq50"] == 51 and counts["kmh_nq100"] == 63
    assert sum(counts.values()) == 165 and len(manifest.groups) == 55
    assert len({g.dir_results for g in manifest.groups}) == len(manifest.groups)  # one leaf dir per group
    assert all(g.strategy_name == "RepresentationStrategy" for g in manifest.groups)
    # A root of its own (ADR 0016): never the first execution's results/campaign_kmh/.
    assert manifest.results_root.endswith("_kmh_fixed") and manifest.results_root != "results/campaign_kmh"
    assert {g.part for g in manifest.groups} == {"kmh_nq10", "kmh_nq50", "kmh_nq100"}


def test_kmh_sweep_aliases_resolve_to_the_reference() -> None:
    manifest = C.load_manifest(SWEEP)
    aliases = manifest.alias_by_id()
    assert set(aliases) == {"kmh/nq100/hier_L3_matched"}
    assert aliases["kmh/nq100/hier_L3_matched"].canonical == "kmh/nq100/hier_L3_ref"
    assert "kmh/nq100/hier_L1_kNQ" not in manifest.group_by_id()
    table_runs = {r["run"] for t in manifest.tables["paper1"]["tables"] for r in t["rows"]}
    assert "kmh/nq100/hier_L3_matched" in table_runs  # the row exists, as an alias
    # every group is consumed by at least one table row (used_by non-empty)
    assert all(g.used_by for g in manifest.groups)


def test_kmh_sweep_parts_filter_and_order() -> None:
    manifest = C.load_manifest(SWEEP)
    assert C.parse_parts("kmh_nq10") == ("kmh_nq10",)
    assert C.parse_parts("kmh_nq10,kmh_nq100") == ("kmh_nq100", "kmh_nq10")  # PARTS order: nq100 first
    jobs_all = C.expand_jobs(manifest, C.parse_parts("all"))
    assert [j.group.part for j in jobs_all][:63] == ["kmh_nq100"] * 63 and jobs_all[-1].group.part == "kmh_nq10"
    jobs = C.expand_jobs(manifest, C.parse_parts("kmh_nq50"))
    assert len(jobs) == 51 and {j.group.part for j in jobs} == {"kmh_nq50"}
    order = [C.PHASE_LABELS[C.phase_index(g, manifest.primary_nq)] for g in manifest.groups]
    assert order == sorted(order, key=C.PHASE_LABELS.index)


@pytest.mark.parametrize("which", ["full_manifest", "micro_manifest"])
def test_run_counts_per_part(which: str, request: pytest.FixtureRequest) -> None:
    manifest: C.Manifest = request.getfixturevalue(which)
    counts = C.count_by_part(C.expand_jobs(manifest))
    assert counts == {"paper1": 189, "upper_bound": 3, "rnhal": 42, "texhal": 30, "texhal_q17": 18,
                      "hier_transfer": 12, "control": 0, "kmh_nq10": 0, "kmh_nq50": 0, "kmh_nq100": 0}
    assert sum(counts.values()) == 294 and len(manifest.groups) == 98


def test_paper1_is_20_strategies_plus_kmh_and_no_rnhal_texhal_presets(full_manifest: C.Manifest) -> None:
    p1 = [g for g in full_manifest.groups if g.part == "paper1"]
    strategies = {g.strategy_name for g in p1}
    assert "SSRAEKmeansHCSampling" not in strategies and "VCTexKmeansHCSampling" not in strategies
    assert set(DEEPALPLUS) <= strategies and len(strategies - {"RepresentationStrategy"}) == 20
    named = {g.id for g in p1 if g.strategy_name != "RepresentationStrategy"}
    assert len(named) == 60  # 59 p1/* groups + shared/random_nq100 (paper-1's RandomSampling@100)
    assert "shared/random_nq100" in named
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
    # ADR 0011: TexHAL canonical Q=17; the reference-hierarchy run is the executed texhal/6_1/rep_q17
    for row in ("6_2/hier_ref", "6_3/stage_full"):
        assert aliases[f"texhal_q17/{row}"].canonical == "texhal/6_1/rep_q17"
    assert aliases["texhal_q17/6_3/stage_no_representation"].canonical == "shared/kmh_nq100"
    q17 = {u["paper"] for u in groups["texhal/6_1/rep_q17"].used_by}
    assert q17 == {"paper2", "paper3"}
    # every group consumed by more than one paper lives under shared/ (except the already-executed
    # texhal/6_1/rep_q17, left in place so the finished runs stay resumable)
    for gid, g in groups.items():
        if len({u["paper"] for u in g.used_by}) > 1 and gid != "texhal/6_1/rep_q17":
            assert gid.startswith("shared/") and "/shared/" in g.dir_results


def test_upper_bound_uses_full_supervised_with_pool_size(full_manifest: C.Manifest, micro_manifest: C.Manifest) -> None:
    ub = full_manifest.group_by_id()["p1/upper_bound"]
    assert (ub.strategy_name, ub.n_init_labeled, ub.n_round) == ("FullSupervised", 7230, 0)  # daninhas_full_fixed pool
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
    assert all(j.group.n_query == 100 for j in jobs[:63]) and jobs[63].group.part == "rnhal"
    assert jobs[-1].group.part == "upper_bound"
    assert [C.PHASE_LABELS[p] for p in dict.fromkeys(phases)] == [
        "paper1_nq100", "rnhal", "texhal", "texhal_q17", "hier_transfer", "paper1_other_nq", "upper_bound"]


@pytest.mark.parametrize("which", ["full_manifest", "micro_manifest"])
def test_paper1_params_per_strategy(which: str, request: pytest.FixtureRequest) -> None:
    manifest: C.Manifest = request.getfixturevalue(which)
    suffix = "_micro" if which == "micro_manifest" else ""
    root = "results/smoke_campaign_fixed" if suffix else "results/campaign_fixed"
    assert manifest.results_root == root
    for group in manifest.groups:
        if group.part != "paper1" or group.strategy_name == "RepresentationStrategy":
            continue
        if group.strategy_name in DROPOUT_FAMILY:
            expected = "params_paper1_mc_dropout"
        elif group.strategy_name == "VAAL":
            expected = "params_paper1_vaal"
        else:
            expected = "params_paper1"
        assert group.params_json == f"files_config/campaign/{expected}{suffix}.json", group.id
        assert group.dir_results.startswith(f"{root}/"), group.id
        assert group.used_by, group.id


def test_full_scale_params_use_the_fixed_dataset_and_vaal_image_32(full_manifest: C.Manifest) -> None:
    from dalmax.config.loader import load_experiment_config

    sweep = C.load_manifest(SWEEP)
    paths = {g.params_json for m in (full_manifest, sweep) for g in m.groups}
    paths |= {a.params_json for m in (full_manifest, sweep) for a in m.aliases}
    for rel in sorted(paths):
        data_dir = json.loads((REPO_ROOT / rel).read_text())["DANINHAS"]["data_dir"]
        assert data_dir == "DATA/daninhas_full_fixed/", rel
    vaal = load_experiment_config(REPO_ROOT / "files_config/campaign/params_paper1_vaal.json", "DANINHAS",
                                  strategy_name="VAAL", seed=1, n_init_labeled=100, n_query=100, n_round=8,
                                  dir_results="x/", device="cpu")
    assert vaal.dataset.strategy_args.vaal.image_size == 32


# --- GPU parts (ADR 0016) ---------------------------------------------------------


@pytest.fixture(scope="module")
def both_full() -> tuple[C.Manifest, C.Manifest]:
    return C.load_manifest(FULL), C.load_manifest(SWEEP)


def _estimates(manifests: tuple[C.Manifest, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for manifest in manifests:
        out.update(C.load_estimates(manifest))
    return out


@pytest.mark.parametrize("paths", [(FULL, SWEEP), (MICRO, SWEEP_MICRO)])
def test_every_group_has_a_gpu_part_and_stage(paths: tuple[Path, Path]) -> None:
    for path in paths:
        manifest = C.load_manifest(path)
        for group in manifest.groups:
            assert group.gpu_part in C.GPU_PARTS and isinstance(group.gpu_stage, int), group.id
        assert set(manifest.required_leaf_files) == set(ROUND_HISTORY_FILES)


def test_gpu_parts_are_balanced_by_the_estimates(both_full: tuple[C.Manifest, C.Manifest]) -> None:
    estimates = _estimates(both_full)
    all_ids = {g.id for m in both_full for g in m.groups}
    assert all_ids <= set(estimates), sorted(all_ids - set(estimates))  # every group is estimated
    hours = {}
    for gpu_part in C.GPU_PARTS:
        plan = C.plan_jobs(both_full, C.parse_selection(gpu_part))
        hours[gpu_part], unknown = C.estimated_hours([j for seg in plan for j in seg.jobs], estimates)
        assert unknown == 0
    total = sum(hours.values())
    assert 90 < total < 105  # 96 GPU hours with the committed estimates
    assert abs(hours["part_1_gpu"] - hours["part_2_gpu"]) / total < 0.05


def test_gpu_parts_cover_every_group_once_and_keep_tables_together(both_full) -> None:
    main, sweep = both_full
    p1 = C.plan_jobs(both_full, C.parse_selection("part_1_gpu"))
    p2 = C.plan_jobs(both_full, C.parse_selection("part_2_gpu"))
    ids1 = {j.group.id for seg in p1 for j in seg.jobs}
    ids2 = {j.group.id for seg in p2 for j in seg.jobs}
    assert not ids1 & ids2 and ids1 | ids2 == {g.id for m in both_full for g in m.groups}
    groups = {g.id: g for m in both_full for g in m.groups}
    aliases = {a.id: a.canonical for m in both_full for a in m.aliases}
    # Every table except paper 1's per-budget KMH-sweep context is inside one GPU part.
    for manifest in both_full:
        for paper_tables in manifest.tables.values():
            for table in paper_tables["tables"]:
                runs = {aliases.get(r["run"], r["run"]) for r in table["rows"] if "run" in r}
                parts = {groups[r].gpu_part for r in runs}
                if table["id"].startswith("paper1_nq") and table["id"] != "paper1_nq100":
                    # paper 1 at nq50/nq10: every row in part_2_gpu except the upper bound (part_1_gpu)
                    assert {groups[r].gpu_part for r in runs - {"p1/upper_bound"}} == {"part_2_gpu"}
                else:
                    assert len(parts) == 1, (table["id"], parts)
    # Papers 2/3 comparisons and 6.3 tables reuse the shared nq100 runs, all in part_1_gpu.
    for gid in ("shared/random_nq100", "shared/kmh_nq100", "shared/texhal_full", "rnhal/6_3/stage_full"):
        assert groups[gid].gpu_part == "part_1_gpu"
    by_budget = {g.n_query: {h.gpu_part for h in main.groups if h.part == "paper1" and h.n_query == g.n_query}
                 for g in main.groups if g.part == "paper1"}
    assert by_budget == {100: {"part_1_gpu"}, 50: {"part_2_gpu"}, 10: {"part_2_gpu"}}
    sweep_parts = {g.part: g.gpu_part for g in sweep.groups}
    assert sweep_parts == {"kmh_nq100": "part_1_gpu", "kmh_nq50": "part_1_gpu", "kmh_nq10": "part_2_gpu"}


def test_gpu_part_plan_runs_cheap_groups_first_sweep_in_the_middle_long_ones_last(both_full) -> None:
    main, sweep = both_full
    for gpu_part in C.GPU_PARTS:
        plan = C.plan_jobs(both_full, C.parse_selection(gpu_part))
        assert [seg.manifest.name for seg in plan] == [main.name, sweep.name, main.name]
        jobs = [j for seg in plan for j in seg.jobs]
        stages = [j.group.gpu_stage for j in jobs]
        assert stages == sorted(stages)
        tail = [j.group.strategy_name for j in plan[-1].jobs]
        assert set(tail) <= set(LONG) and tail[-1] == "AdversarialDeepFool"
        assert not [j for seg in plan[:-1] for j in seg.jobs if j.group.strategy_name in LONG]
        if gpu_part == "part_1_gpu":
            assert jobs[0].group.id == "shared/random_nq100"


def test_parse_selection_gpu_parts_and_mixing() -> None:
    assert C.parse_selection("part_2_gpu") == C.Selection(C.PARTS, ("part_2_gpu",))
    assert C.parse_selection("part_2_gpu,part_1_gpu").gpu_parts == C.GPU_PARTS
    assert C.parse_selection("rnhal").gpu_parts is None
    with pytest.raises(C.CampaignError, match="cannot be mixed"):
        C.parse_selection("part_1_gpu,rnhal")
    with pytest.raises(C.CampaignError):
        C.parse_parts("part_1_gpu")  # semantic parts only


def test_cli_list_gpu_part_prints_estimates(capsys: pytest.CaptureFixture[str]) -> None:
    manifests = f"{FULL},{SWEEP}"
    assert C.main(["list", "--manifest", manifests, "--part", "part_2_gpu"]) == 0
    out = capsys.readouterr().out
    assert "Execution plan (3 segments)" in out and "GPU parts (ADR 0016):" in out
    combined = out.split("Combined:")[1]
    assert "part_2_gpu     177 runs (59 run groups)  est" in combined
    assert "part_1_gpu       0 runs" in combined


def test_cli_run_gpu_part_dry_run_logs_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = C.main(["run", "--manifest", f"{MICRO},{SWEEP_MICRO}", "--part", "part_2_gpu", "--seeds", "1",
                   "--dry-run", "--no-skip-existing", "--device", "cpu"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("CAMPAIGN START") == 3 and "DRY-RUN" in out


def test_hier_transfer_part_is_four_no_resampling_groups(full_manifest: C.Manifest) -> None:
    groups = [g for g in full_manifest.groups if g.part == "hier_transfer"]
    assert sorted(g.id for g in groups) == [
        "rnhal/6_2/hier_L2_matched__no_resampling", "rnhal/6_2/hier_L3__no_resampling",
        "texhal_q17/6_2/hier_L2_matched__no_resampling", "texhal_q17/6_2/hier_L3__no_resampling"]
    assert all(g.seeds == (1, 2, 3) or list(g.seeds) == [1, 2, 3] for g in groups)
    assert all(g.used_by for g in groups)
    assert C.parse_parts("hier_transfer,texhal_q17") == ("texhal_q17", "hier_transfer")


def test_parse_parts_and_seed_subset(full_manifest: C.Manifest) -> None:
    assert C.parse_parts("all") == C.PARTS
    assert C.parse_parts("texhal_q17,rnhal") == ("rnhal", "texhal_q17")
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


def _write_leaf(leaf: Path, *, omit: tuple[str, ...] = (), initial: str | None = None, n_round: int = 1,
                round_files: bool = False, order: tuple[str, str] | None = ("train0", "test0"),
                initial_paths: list[str] | None = None, query_paths: list[list[str]] | None = None) -> None:
    """A fake leaf. `order` = the `data_order` hashes of run_metadata.json (None: a pre-KI-47 leaf);
    `initial_paths` (with `round_files`) = the round-0 image paths of round_queries.json."""
    leaf.mkdir(parents=True, exist_ok=True)
    payload = {"all_acc": [0.5, 0.6], "all_f1_score": [0.5, 0.6], "all_f1_macro": [0.4, 0.5],
               "rounds": list(range(n_round + 1))}
    meta: dict = {"config": {}, "git_commit": "abc"}
    if order is not None:
        meta["data_order"] = {"contract": "sorted-names-v1", "train_paths_hash": order[0], "test_paths_hash": order[1]}
    files = {
        "results.json": json.dumps(payload),
        "predictions.csv": "Image Index,Real Class,Predicted Class,Correct,Path\n0,a,a,True,x\n",
        "run_metadata.json": json.dumps(meta),
        "log-dalmax.log": f"Initial labeled idxs (sorted): {initial or '[1, 2, 3]'}\n",
    }
    if round_files:
        files["round_predictions.json"] = json.dumps({"predictions": [[0]] * (n_round + 1)})
        if initial_paths is None:
            rounds = [{"round": r} for r in range(n_round + 1)]
        else:
            queries = query_paths or [[] for _ in range(n_round)]
            rounds = [{"round": 0, "kind": "initial", "pool_idxs": json.loads(initial or "[1, 2, 3]"),
                       "paths": initial_paths}]
            rounds += [{"round": r, "kind": "query", "pool_idxs": list(range(len(queries[r - 1]))),
                        "paths": queries[r - 1]} for r in range(1, n_round + 1)]
        files["round_queries.json"] = json.dumps({"rounds": rounds})
    for name in (*REQUIRED_FILES, *(ROUND_HISTORY_FILES if round_files else ())):
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
    assert len(kept) == len(jobs) - len(dropped) == 294 - 18
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
    assert "TOTAL" in out and "92 runs" in out  # 98 micro seed-1 jobs - 2 strategies x 3 n_query
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


# --- required per-round files (ADR 0016) ------------------------------------------------


def test_required_leaf_files_make_old_style_leaves_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = replace(_tiny_manifest(tmp_path), required_leaf_files=ROUND_HISTORY_FILES)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    g = manifest.groups[0]
    _write_leaf(_leaf_of(g, 1))  # the artifact set of the earlier trees, no per-round files
    _write_leaf(_leaf_of(g, 2), round_files=True)
    job1, job2 = C.Job(g, 1), C.Job(g, 2)
    assert C.verify_job(job1).status == "OK"  # without the requirement: old trees still verify
    status = C.verify_job(job1, required_extra=manifest.required_leaf_files)
    assert status.status == "INCOMPLETE" and set(status.missing_files) == set(ROUND_HISTORY_FILES)
    assert C.verify_job(job2, required_extra=manifest.required_leaf_files).status == "OK"
    assert not C.job_done(job1, manifest.required_leaf_files) and C.job_done(job2, manifest.required_leaf_files)
    calls: list[list[str]] = []
    summary = C.run_jobs([job1, job2], manifest, device="cpu", executor=lambda cmd, env: calls.append(cmd) or 0,
                         echo=lambda _: None)
    assert (summary.executed, summary.skipped) == (1, 1) and len(calls) == 1  # the old-style leaf is re-run


def test_round_history_with_a_wrong_round_count_warns(tmp_path: Path) -> None:
    from dalmax.reporting.leaf_check import check_leaf

    manifest = _tiny_manifest(tmp_path)
    leaf = _leaf_of(manifest.groups[0], 1)
    _write_leaf(leaf, round_files=True)
    (leaf / "round_queries.json").write_text(json.dumps({"rounds": [{"round": 0}]}))  # NR_1 needs 2 rounds
    missing, warnings = check_leaf(leaf, ROUND_HISTORY_FILES)
    assert not missing and any("round_queries.json: 1 rounds" in w for w in warnings)
    with pytest.raises(ValueError, match="unknown extra"):
        check_leaf(leaf, ("nope.json",))


def test_validate_rejects_unknown_required_files_and_half_gpu_fields(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    with pytest.raises(C.CampaignError, match="required_leaf_files"):
        C.validate_manifest(replace(manifest, required_leaf_files=("nope.json",)))
    half = replace(manifest.groups[0], gpu_part="part_1_gpu")
    with pytest.raises(C.CampaignError, match="set together"):
        C.validate_manifest(replace(manifest, groups=(half, manifest.groups[1])))
    bad = replace(manifest.groups[0], gpu_part="part_3_gpu", gpu_stage=1)
    with pytest.raises(C.CampaignError, match="gpu_part"):
        C.validate_manifest(replace(manifest, groups=(bad, manifest.groups[1])))


# --- signature: strategy_args reduced to the strategy's own block (ADR 0015/0016) ---------


def test_signature_ignores_other_strategies_args_blocks() -> None:
    args = ("DANINHAS",)
    base, vaal = "files_config/campaign/params_paper1.json", "files_config/campaign/params_paper1_vaal.json"
    for strategy in ("RandomSampling", "LeastConfidence", "CEALSampling", "WAAL", "AdversarialDeepFool"):
        # params_paper1_vaal.json differs from params_paper1.json only by the VAAL block
        assert (C.effective_signature(base, *args, strategy, 100, 100, 8)
                == C.effective_signature(vaal, *args, strategy, 100, 100, 8)), strategy
    # VAAL reads its block: image_size 32 vs the default 64 is a different computation.
    assert (C.effective_signature(base, *args, "VAAL", 100, 100, 8)
            != C.effective_signature(vaal, *args, "VAAL", 100, 100, 8))
    sig = json.loads(C.effective_signature(vaal, *args, "VAAL", 100, 100, 8))
    assert list(sig["dataset"]["strategy_args"]) == ["vaal"] and sig["dataset"]["strategy_args"]["vaal"]["image_size"] == 32
    assert "strategy_args" not in json.loads(C.effective_signature(vaal, *args, "RandomSampling", 100, 100, 8))["dataset"]


# --- KI-47: seed audit by image paths, data-order audit -------------------------------------


IMAGES = ["DATA/d/train/c0/a.png", "DATA/d/train/c0/b.png", "DATA/d/train/c1/c.png"]


def test_seed_audit_fails_on_different_images_under_equal_indices(tmp_path: Path) -> None:
    """The KI-47 failure mode: the same seed selects the same pool INDICES, which name other images
    under another data order. Indices alone pass; the round-0 paths of round_queries.json fail."""
    manifest = _tiny_manifest(tmp_path)
    g_random, g_margin = manifest.groups
    _write_leaf(_leaf_of(g_random, 1), round_files=True, initial="[1, 2, 3]", initial_paths=IMAGES)
    other_images = ["DATA/d/train/c0/x.png", "DATA/d/train/c0/b.png", "DATA/d/train/c1/c.png"]
    _write_leaf(_leaf_of(g_margin, 1), round_files=True, initial="[1, 2, 3]", initial_paths=other_images)
    jobs = C.expand_jobs(manifest, seeds=[1])

    audit = C.audit_seed_consistency(jobs)[0]
    assert audit.status == "FAIL" and audit.offenders == ("p1/MarginSampling/nq5",)
    assert audit.basis == "image paths"
    assert "compared by image paths" in C.render_seed_audit([audit])

    # Same images under another dataset location (absolute path, other machine): PASS.
    moved = ["/content/dalmax/" + p for p in reversed(IMAGES)]
    _write_leaf(_leaf_of(g_margin, 1), round_files=True, initial="[1, 2, 3]", initial_paths=moved)
    assert C.audit_seed_consistency(jobs)[0].status == "PASS"


def test_seed_audit_falls_back_to_log_indices_for_older_leaves(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    g_random, g_margin = manifest.groups
    _write_leaf(_leaf_of(g_random, 1), round_files=True, initial="[1, 2, 3]", initial_paths=IMAGES)
    _write_leaf(_leaf_of(g_margin, 1), initial="[1, 2, 3]")  # no round_queries.json: log indices only
    audit = C.audit_seed_consistency(C.expand_jobs(manifest, seeds=[1]))[0]
    assert audit.status == "PASS" and audit.basis == "image paths + pool indices"
    _write_leaf(_leaf_of(g_margin, 1), initial="[1, 2, 4]")
    assert C.audit_seed_consistency(C.expand_jobs(manifest, seeds=[1]))[0].status == "FAIL"


def test_read_initial_set_and_round_paths(tmp_path: Path) -> None:
    _write_leaf(tmp_path, round_files=True, initial="[4, 8]", initial_paths=["x/train/c/b.png", "x/train/c/a.png"],
                query_paths=[["x/train/c/z.png"]])
    assert C.read_initial_set(tmp_path) == C.InitialSet([4, 8], ["c/a.png", "c/b.png"])
    assert C.read_round_paths(tmp_path) == [["c/a.png", "c/b.png"], ["c/z.png"]]
    (tmp_path / "round_queries.json").write_text("{not json")
    assert C.read_initial_set(tmp_path) == C.InitialSet([4, 8], None)  # falls back to the log line
    assert C.read_round_paths(tmp_path) is None


def test_data_order_audit_pass_fail_warn_and_exit_codes(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    g_random, g_margin = manifest.groups
    for seed in (1, 2):
        _write_leaf(_leaf_of(g_random, seed))
        _write_leaf(_leaf_of(g_margin, seed))
    jobs = C.expand_jobs(manifest)
    statuses = [C.verify_job(j) for j in jobs]
    seed_audits = C.audit_seed_consistency(jobs)
    audit = C.audit_data_order(jobs)
    assert [(a.dataset, a.status, a.n_runs) for a in audit] == [("daninhas_micro", "PASS", 4)]
    assert C.verify_exit_code(statuses, seed_audits, audit) == 0

    _write_leaf(_leaf_of(g_margin, 2), order=("train1", "test0"))  # one leaf indexed another train order
    audit = C.audit_data_order(jobs)
    assert audit[0].status == "FAIL" and len(audit[0].variants) == 2
    text = C.render_data_order_audit(audit)
    assert "daninhas_micro: FAIL" in text and "train train1 / test test0: 1 runs" in text
    assert C.verify_exit_code(statuses, seed_audits, audit) == 1

    _write_leaf(_leaf_of(g_margin, 2), order=None)  # a leaf written before KI-47: WARN
    audit = C.audit_data_order(jobs)
    assert audit[0].status == "WARN" and audit[0].missing == ("p1/MarginSampling/nq5 seed=2",)
    assert C.verify_exit_code(statuses, seed_audits, audit) == 2


# --- N5: intentional duplicates across the two manifests ------------------------------------


def test_the_only_duplicates_across_manifests_are_the_three_reference_kmh_pairs(both_full) -> None:
    pairs = {(a.id, b.id) for a, b in C.cross_manifest_duplicates(both_full)}
    assert pairs == {
        ("shared/kmh_nq100", "kmh/nq100/hier_L3_ref"),
        ("p1/KMH/nq50", "kmh/nq50/hier_L3_ref"),
        ("p1/KMH/nq10", "kmh/nq10/hier_L3_ref"),
    }


def test_duplicate_pair_check_same_different_and_missing(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    a, b = manifest.groups
    _write_leaf(_leaf_of(a, 1), round_files=True, initial_paths=IMAGES, query_paths=[["DATA/d/train/c1/q.png"]])
    _write_leaf(_leaf_of(b, 1), round_files=True, initial_paths=IMAGES, query_paths=[["/abs/train/c1/q.png"]])
    _write_leaf(_leaf_of(a, 2), round_files=True, initial_paths=IMAGES, query_paths=[["DATA/d/train/c1/q.png"]])
    _write_leaf(_leaf_of(b, 2), round_files=True, initial_paths=IMAGES, query_paths=[["DATA/d/train/c1/r.png"]])
    checks = {c.seed: c for c in C.check_duplicate_pairs([(a, b)], {1, 2})}
    assert checks[1].status == "SAME" and checks[2].status == "DIFFERENT"
    assert "first differing round 1" in checks[2].detail
    (_leaf_of(b, 2) / "round_queries.json").unlink()
    assert C.check_duplicate_pairs([(a, b)], {2})[0].status == "N/A"
    assert "SAME" in C.render_duplicate_pairs(list(checks.values()))


# --- N1: free-space gate --------------------------------------------------------------------


def test_space_checks_group_roots_by_file_system(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    plan = C.plan_jobs([manifest], C.Selection())
    checks = C.space_checks(plan, free_space=lambda p: 10**9, device_of=lambda p: 7, run_bytes=100)
    assert len(checks) == 1 and checks[0].remaining_runs == 4 and checks[0].needed == 480 and checks[0].ok
    _write_leaf(_leaf_of(manifest.groups[0], 1))  # a complete leaf is not "remaining"
    assert C.space_checks(plan, free_space=lambda p: 10**9, device_of=lambda p: 7)[0].remaining_runs == 3
    short = C.space_checks(plan, free_space=lambda p: 0, device_of=lambda p: 7)
    assert not short[0].ok
    assert "NOT ENOUGH" in C.render_space_checks(short) and "3 remaining runs" in C.render_space_checks(short)


def test_run_refuses_to_start_without_space(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("no job may start when the free space is short")

    monkeypatch.setattr(C, "free_bytes", lambda path: 0)
    monkeypatch.setattr(C, "run_jobs", forbidden)
    monkeypatch.setattr(C, "job_done", lambda job, required_extra=(): False)  # independent of local smoke trees
    code = C.main(["run", "--micro", "--part", "rnhal", "--seeds", "1", "--device", "cpu", "--no-skip-existing"])
    assert code == C.EXIT_NO_SPACE == 4
    err = capsys.readouterr().err
    assert "NOT ENOUGH" in err and "nothing was started" in err
    assert C.main(["space", "--micro", "--part", "rnhal", "--seeds", "1"]) == 4


def test_run_jobs_stops_on_low_space_and_on_a_no_space_exit(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    jobs = C.expand_jobs(manifest)
    calls: list[list[str]] = []

    def executor(cmd: list[str], env: dict[str, str]) -> int:
        calls.append(cmd)
        return C.NO_SPACE_EXIT_CODE

    summary = C.run_jobs(jobs, manifest, device="cpu", executor=executor, echo=lambda _: None,
                         free_space=lambda p: 10**12)
    assert summary.stopped_no_space and summary.failed == 1 and len(calls) == 1  # stopped, not continued
    assert "STOP no space" in (tmp_path / "campaign.log").read_text()

    calls.clear()
    free = iter([10**12, 10])  # enough for the first run, then the disk is full

    def ok_executor(cmd: list[str], env: dict[str, str]) -> int:
        calls.append(cmd)
        return 0

    summary = C.run_jobs(jobs, manifest, device="cpu", executor=ok_executor, echo=lambda _: None,
                         free_space=lambda p: next(free), skip_existing=False)
    assert summary.stopped_no_space and summary.executed == 1 and len(calls) == 1


# --- N4: pilot --------------------------------------------------------------------------------


def test_pilot_groups_per_gpu_part(both_full) -> None:
    part1 = C.pilot_group_ids(C.selected_groups(both_full, C.parse_selection("part_1_gpu")))
    part2 = C.pilot_group_ids(C.selected_groups(both_full, C.parse_selection("part_2_gpu")))
    assert part1 == ("shared/random_nq100", "p1/VAAL/nq100", "p1/WAAL/nq100")
    assert part2 == ("p1/RandomSampling/nq50", "p1/VAAL/nq50", "p1/WAAL/nq50")  # no nq100 in part 2
    with pytest.raises(C.CampaignError, match="no VAAL/WAAL group"):
        C.pilot_group_ids(C.selected_groups(both_full, C.parse_selection("rnhal")))


def test_only_group_restricts_and_fails_fast(both_full, capsys) -> None:
    plan = C.plan_jobs(both_full, C.parse_selection("part_1_gpu"), seeds=[1])
    groups = C.selected_groups(both_full, C.parse_selection("part_1_gpu"))
    kept = C.restrict_plan(plan, ["p1/VAAL/nq100"], [g.id for g in groups])
    assert [(j.group.id, j.seed) for seg in kept for j in seg.jobs] == [("p1/VAAL/nq100", 1)]
    with pytest.raises(C.CampaignError, match="not in the selected groups"):
        C.restrict_plan(plan, ["p1/VAAL/nq50"], [g.id for g in groups])  # a group of the other GPU part
    manifests = f"{FULL},{SWEEP}"
    assert C.main(["list", "--manifest", manifests, "--part", "part_1_gpu", "--only-group", "nope"]) == 2
    assert C.main(["list", "--manifest", manifests, "--part", "part_2_gpu", "--pilot"]) == 0
    out = capsys.readouterr().out
    assert "p1/VAAL/nq50" in out and "TOTAL            3 runs (3 run groups)" in out.split("Combined:")[1]
    with pytest.raises(SystemExit):
        C.main(["list", "--pilot", "--only-group", "x"])  # mutually exclusive


def test_full_run_after_the_pilot_skips_the_completed_pilot_leaves(tmp_path: Path) -> None:
    def group(gid: str, strategy: str) -> C.RunGroup:
        return C.RunGroup(id=gid, part="paper1", params_json="files_config/campaign/params_paper1_micro.json",
                          strategy_name=strategy, n_query=5, n_init_labeled=10, n_round=1, seeds=(1, 2),
                          dir_results=str(tmp_path / gid) + "/", used_by=())

    groups = tuple(group(f"p1/{s}/nq5", s) for s in ("RandomSampling", "VAAL", "WAAL", "MarginSampling"))
    manifest = C.Manifest(name="tiny", dataset_name="DANINHAS", results_root=str(tmp_path), seeds=(1, 2),
                          primary_nq=5, groups=groups, aliases=(), tables={})
    by_strategy = {g.strategy_name: g for g in groups}
    executed: list[tuple[str, int]] = []

    def executor(cmd: list[str], env: dict[str, str]) -> int:
        g = by_strategy[cmd[cmd.index("--strategy_name") + 1]]
        seed = int(cmd[cmd.index("--seed") + 1])
        executed.append((g.id, seed))
        _write_leaf(_leaf_of(g, seed))
        return 0

    pilot_ids = C.pilot_group_ids(groups)
    assert pilot_ids == ("p1/RandomSampling/nq5", "p1/VAAL/nq5", "p1/WAAL/nq5")
    pilot_plan = C.restrict_plan(C.plan_jobs([manifest], C.Selection(), seeds=[1]), pilot_ids, pilot_ids)
    for seg in pilot_plan:
        C.run_jobs(seg.jobs, manifest, device="cpu", executor=executor, echo=lambda _: None,
                   free_space=lambda p: 10**12)
    assert sorted(executed) == sorted((gid, 1) for gid in pilot_ids)

    executed.clear()
    summary = C.run_jobs(C.expand_jobs(manifest), manifest, device="cpu", executor=executor, echo=lambda _: None,
                         free_space=lambda p: 10**12)
    assert summary.skipped == 3 and summary.executed == 5
    assert not set(executed) & {(gid, 1) for gid in pilot_ids}
