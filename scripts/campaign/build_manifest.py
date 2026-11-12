"""Generate `files_config/campaign/manifest.json`, `manifest_micro.json`, `manifest_control_ki43.json`
and the KMH sweep manifests `manifest_kmh_sweep.json` / `manifest_kmh_sweep_micro.json`.

Since ADR 0016 they describe the clean re-execution of everything on `DATA/daninhas_full_fixed`
(results under `results/campaign_fixed/` and `results/campaign_kmh_fixed/`), with every group of the
two full-scale manifests also assigned to one of the two GPU parts (`part_1_gpu`, `part_2_gpu`, one
per Colab account). The manifests of the first execution (dataset with duplicates, KI-46) are in the
git history (commit d668673 and earlier).

The manifests are committed artifacts (single source of truth, see
`.specs/experiments/campaign.md`); this script is how they are produced,
and `tests/test_campaign.py` fails if the committed files drift from what it
emits. Run from the repo root:

    poetry run python scripts/campaign/build_manifest.py            # rewrite all
    poetry run python scripts/campaign/build_manifest.py --check    # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dalmax.campaign import GPU_PARTS, Alias, derive_used_by  # noqa: E402
from dalmax.reporting.leaf_check import ROUND_HISTORY_FILES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kmh_sweep_configs as sweep  # noqa: E402

SEEDS = [1, 2, 3]
# Paper 1: the 12 classical strategies (no RNHAL/TexHAL -- those belong to papers 3/2).
CLASSICAL_STRATEGIES = [
    "RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling",
    "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout",
    "KMeansSampling", "KCenterGreedy", "BALDDropout", "AdversarialBIM",
    "AdversarialDeepFool",
]
# Paper 1 since ADR 0016: the eight DeepAL+ ports of ADR 0015, at the same three budgets.
DEEPALPLUS_STRATEGIES = [
    "VarRatio", "MeanSTD", "KCenterGreedyPCA", "BadgeSampling", "CEALSampling", "LossPredictionLoss",
    "VAAL", "WAAL",
]
PAPER1_STRATEGIES = [*CLASSICAL_STRATEGIES, *DEEPALPLUS_STRATEGIES]
# KMH = plain hierarchical k-means on ImageNet ResNet-50 features (no SSRAE/VCTex);
# interpretation CONFIRMED by the user 2026-09-29 (see .specs/experiments/campaign.md).
KMH = "KMH"
KMH_LABEL = "KMH (hierarchical k-means, ImageNet ResNet-50 features)"
# Paper-1 methods eligible for "best paper-1 method @ nq=100": the 20 strategies + KMH.
PAPER1_METHODS = [*PAPER1_STRATEGIES, KMH]
PAPER1_BUDGETS = (10, 50, 100)
# The dropout family trains with a dropout layer before the classifier (KI-41, ADR 0014):
# the four MC-dropout strategies and the two DeepAL+ ones that sample it (VarRatio, MeanSTD).
DROPOUT_STRATEGIES = [
    "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout", "BALDDropout", "VarRatio", "MeanSTD",
]

# --- GPU parts (ADR 0016) ----------------------------------------------------------------------------
# A second, orthogonal partition of BOTH full-scale manifests (main + KMH sweep) into two halves of about
# equal estimated GPU time (files_config/campaign/gpu_time_estimates.json), one per Colab account. A group
# never splits; groups read together in one table stay in one part where the balance allows:
#   part_1_gpu = paper 1 at nq100 (with the shared random/KMH runs), the upper bound, every ablation part
#                of papers 2/3 (all their tables and comparisons are at nq100), the KMH sweep at nq100 and
#                at nq50 (the nq50 sweep is the balancing item: its own tables stay in one part);
#   part_2_gpu = paper 1 at nq50 and nq10, the KMH sweep at nq10.
PART_1_GPU, PART_2_GPU = GPU_PARTS
PAPER1_GPU_PART: dict[int, str] = {100: PART_1_GPU, 50: PART_2_GPU, 10: PART_2_GPU}
SWEEP_GPU_PART: dict[int, str] = {100: PART_1_GPU, 50: PART_1_GPU, 10: PART_2_GPU}
ABLATION_GPU_PART = PART_1_GPU
UPPER_BOUND_GPU_PART = PART_1_GPU
# Execution stage inside a GPU part (both manifests share the numbering; `campaign run --part part_N_gpu`
# orders by it): cheap table-critical groups first, the very long ones last, so that an exhausted credit
# budget leaves complete tables.
STAGE_PAPER1 = 1  # paper-1 strategies other than the long ones, KMH, the upper bound
STAGE_ABLATIONS = 2  # rnhal, texhal, texhal_q17, hier_transfer
STAGE_KMH_SWEEP = 3  # the KMH sweep manifest
LONG_STRATEGY_STAGE: dict[str, int] = {"AdversarialBIM": 4, "VAAL": 5, "WAAL": 5, "AdversarialDeepFool": 6}
GPU_TIME_ESTIMATES = "files_config/campaign/gpu_time_estimates.json"


@dataclass(frozen=True)
class Scale:
    """Everything that differs between the full-scale and the micro manifest."""

    name: str
    results_root: str
    sweep_results_root: str
    paper1_params: str
    paper1_mc_dropout_params: str
    paper1_vaal_params: str
    kmh_params: str
    abl_subdir: str  # "" or "micro/"
    nq_by_paper_nq: dict[int, int]  # paper nq -> actual --n_query
    n_init: int
    n_round: int
    upper_bound_n_init: int
    gpu_time_estimates: str  # "" = none (the micro mirror: its costs are not the GPU ones)


# Results go to NEW roots (ADR 0016): nothing collides with the executed trees results/campaign/ and
# results/campaign_kmh/ (dataset with duplicates, KI-46), which stay untouched. Upper bound n_init = the
# pool of DATA/daninhas_full_fixed (7,230 training images).
FULL = Scale(
    "campaign_fixed", "results/campaign_fixed", "results/campaign_kmh_fixed",
    "files_config/campaign/params_paper1.json", "files_config/campaign/params_paper1_mc_dropout.json",
    "files_config/campaign/params_paper1_vaal.json", "files_config/campaign/params_kmh.json", "",
    {10: 10, 50: 50, 100: 100}, 100, 8, 7230, GPU_TIME_ESTIMATES,
)
MICRO = Scale(
    "campaign_fixed_micro", "results/smoke_campaign_fixed", "results/smoke_campaign_kmh_fixed",
    "files_config/campaign/params_paper1_micro.json", "files_config/campaign/params_paper1_mc_dropout_micro.json",
    "files_config/campaign/params_paper1_vaal_micro.json", "files_config/campaign/params_kmh_micro.json",
    "micro/",
    {10: 3, 50: 4, 100: 5}, 10, 1, 806, "",
)


def paper1_params(scale: Scale, strategy: str) -> str:
    """Params file of a paper-1 strategy: the dropout family, VAAL, or the common file."""
    if strategy in DROPOUT_STRATEGIES:
        return scale.paper1_mc_dropout_params
    if strategy == "VAAL":
        return scale.paper1_vaal_params
    return scale.paper1_params


def manifest_header(scale: Scale) -> dict[str, Any]:
    """Keys shared by the main and the sweep manifest of one scale (ADR 0016)."""
    header: dict[str, Any] = {"required_leaf_files": list(ROUND_HISTORY_FILES)}
    if scale.gpu_time_estimates:
        header["gpu_time_estimates"] = scale.gpu_time_estimates
    return header


# (id suffix, params basename, table label) -- ablation rows per study, per method.
RNHAL_61 = [("rep_spatial", "Spatial-only"), ("rep_spectral", "Spectral-only")]
TEXHAL_61 = [("rep_q5", "Q=5"), ("rep_q13", "Q=13"), ("rep_q17", "Q=17 (TexHAL)")]
HIER_COMMON = [
    ("hier_L1", "L=1, k=[50]"),
    ("hier_L2a", "L=2, k=[300,100]"),
    ("hier_L2b", "L=2, k=[100,50]"),
    ("hier_L3", "L=3, k=[300,100,50]"),
    ("hier_L4", "L=4, k=[300,100,50,25]"),
]
HIER_NEW_RNHAL = [
    ("hier_L1_k100", "L=1, k=[100]"),
    ("hier_L1_k200", "L=1, k=[200]"),
    ("hier_L1_k600", "L=1, k=[600]"),
    ("hier_L2_k200_100", "L=2, k=[200,100]"),
    ("hier_L4_k800_600_200_100", "L=4, k=[800,600,200,100]"),
]
# §6.2 row order for the tables (grouped by L); "ref" is the reference-hierarchy alias row.
HIER_ORDER_RNHAL = [
    "hier_L1", "hier_L1_k100", "hier_L1_k200", "hier_L1_k600",
    "hier_L2a", "hier_L2b", "hier_L2_k200_100",
    "hier_L3", "ref",
    "hier_L4", "hier_L4_k800_600_200_100",
]
HIER_ORDER_TEXHAL = ["hier_L1", "hier_L2a", "hier_L2b", "hier_L3", "ref", "hier_L4"]
# Rows appended to the 6.2 tables of the canonical methods (RNHAL, TexHAL Q=17) by the hier_transfer part.
HIER_TRANSFER_ORDER = ["hier_L2_matched__no_resampling", "hier_L3__no_resampling"]
HIER_TRANSFER_LABELS = {
    "hier_L2_matched__no_resampling": "L=2, k=[600,100], no resampling (KMH sweep winner)",
    "hier_L3__no_resampling": "L=3, k=[600,200,100] (reference), no resampling",
}
HIER_TRANSFER = list(HIER_TRANSFER_LABELS)
REF_LABEL = "L=3, k=[600,200,100] (reference)"


def build(scale: Scale) -> dict[str, Any]:
    abl = f"files_config/ablations/%s/{scale.abl_subdir}"
    root = scale.results_root
    primary = scale.nq_by_paper_nq[100]
    kmh_params = scale.kmh_params
    groups: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    alias_notes: dict[str, list[str]] = {}

    def add_group(gid: str, part: str, params: str, strategy: str, nq: int, nil: int, nr: int,
                  dir_results: str, note: str = "", *, gpu_part: str, stage: int) -> None:
        groups.append({
            "id": gid, "part": part, "gpu_part": gpu_part,
            "gpu_stage": LONG_STRATEGY_STAGE.get(strategy, stage), "params_json": params,
            "strategy_name": strategy, "n_query": nq, "n_init_labeled": nil, "n_round": nr, "seeds": list(SEEDS),
            "dir_results": dir_results, "used_by": [], "note": note,
        })

    def add_alias(aid: str, canonical: str, params: str, reason: str,
                  strategy: str = "RepresentationStrategy") -> None:
        aliases.append({"id": aid, "canonical": canonical, "params_json": params,
                        "strategy_name": strategy, "reason": reason})
        alias_notes.setdefault(canonical, []).append(aid)

    # Ownership rule: a group consumed by MORE THAN ONE paper lives under shared/
    # (dir results/<root>/shared/<name>/); every other group lives under its paper.
    # `used_by` (derived from the tables) lists all consumers either way.
    shared_note = "Consumed by several papers (see used_by): ONE run, every table points here."

    # --- paper 1 (20 strategies + KMH) x nq {100, 50, 10} ----------------------
    # First job of the whole campaign: the cheapest run (a quick GPU/environment sanity check).
    p1_stage = {"gpu_part": PAPER1_GPU_PART[100], "stage": STAGE_PAPER1}
    add_group("shared/random_nq100", "paper1", scale.paper1_params, "RandomSampling", primary,
              scale.n_init, scale.n_round, f"{root}/shared/random_nq100/", shared_note, **p1_stage)
    for paper_nq in (100, 50, 10):
        nq = scale.nq_by_paper_nq[paper_nq]
        at_nq = {"gpu_part": PAPER1_GPU_PART[paper_nq], "stage": STAGE_PAPER1}
        for strat in PAPER1_STRATEGIES:
            if paper_nq == 100 and strat == "RandomSampling":
                continue  # shared/random_nq100 (above): baseline of papers 1, 2 and 3
            add_group(f"p1/{strat}/nq{paper_nq}", "paper1", paper1_params(scale, strat), strat, nq,
                      scale.n_init, scale.n_round, f"{root}/paper1/nq{paper_nq}/", **at_nq)
        if paper_nq == 100:
            add_group("shared/kmh_nq100", "paper1", kmh_params, "RepresentationStrategy", nq,
                      scale.n_init, scale.n_round, f"{root}/shared/kmh_nq100/",
                      "KMH @ n_query=100 = hierarchical k-means on ImageNet ResNet-50 features (reference "
                      "hierarchy [600,200,100]) = the 'w/o representation module' row of papers 2 and 3. "
                      + shared_note, **at_nq)
        else:
            add_group(f"p1/{KMH}/nq{paper_nq}", "paper1", kmh_params, "RepresentationStrategy", nq,
                      scale.n_init, scale.n_round, f"{root}/paper1/nq{paper_nq}/",
                      "KMH: hierarchical k-means selection on ImageNet ResNet-50 features (reference "
                      "hierarchy [600,200,100]), no SSRAE/VCTex.", **at_nq)
    add_group("p1/upper_bound", "upper_bound", scale.paper1_params, "FullSupervised", primary,
              scale.upper_bound_n_init, 0, f"{root}/paper1/upper_bound/",
              "Upper bound (not active learning): train once on the ENTIRE pool "
              "(n_init_labeled == pool size, n_round 0), evaluate on the fixed test set.",
              gpu_part=UPPER_BOUND_GPU_PART, stage=STAGE_PAPER1)

    # --- rnhal / texhal ablation groups (full rows are ONE canonical run) -----
    def ablation_groups(method: str, study_rows: dict[str, list[str]], part: str | None = None) -> None:
        # `method` = config folder + results dir + group-id prefix; `part` defaults to it.
        for study, cfgs in study_rows.items():
            for cfg in cfgs:
                add_group(f"{method}/{study}/{cfg}", part or method, (abl % method) + f"{cfg}.json",
                          "RepresentationStrategy", primary, scale.n_init, scale.n_round,
                          f"{root}/{method}/{study}/{cfg}/", gpu_part=ABLATION_GPU_PART, stage=STAGE_ABLATIONS)

    ablation_groups("rnhal", {
        "6_1": [c for c, _ in RNHAL_61],
        "6_2": [c for c, _ in HIER_COMMON + HIER_NEW_RNHAL],
        "6_3": ["stage_full", "stage_no_hierarchy"],
    })
    ablation_groups("texhal", {
        "6_1": [c for c, _ in TEXHAL_61],
        "6_2": [c for c, _ in HIER_COMMON],
        "6_3": ["stage_no_hierarchy"],
    })
    # TexHAL canonical Q=17 (ADR 0011): 6.2 hierarchy + 6.3 no-hierarchy re-run with a single RAE.
    # The reference hierarchy with Q=17 IS texhal/6_1/rep_q17 (aliases below); nothing else is re-run.
    ablation_groups("texhal_q17", {
        "6_2": [c for c, _ in HIER_COMMON],
        "6_3": ["stage_no_hierarchy"],
    })
    # Hierarchy transfer (ADR 0013): the KMH-sweep nq100 winner shape (L=2 [600,100], no resampling) and the
    # reference shape without resampling, with each paper's learned representation (SSRAE Q=13, VCTex Q=17).
    ablation_groups("rnhal", {"6_2": HIER_TRANSFER}, "hier_transfer")
    ablation_groups("texhal_q17", {"6_2": HIER_TRANSFER}, "hier_transfer")
    # TexHAL 'full' is consumed by paper 2 (6.1/6.2/6.3 + comparison) AND paper 3 (comparison).
    add_group("shared/texhal_full", "texhal", (abl % "texhal") + "stage_full.json",
              "RepresentationStrategy", primary, scale.n_init, scale.n_round,
              f"{root}/shared/texhal_full/",
              "TexHAL 'full' (VCTex q=[5,17] + reference hierarchy). " + shared_note,
              gpu_part=ABLATION_GPU_PART, stage=STAGE_ABLATIONS)

    # --- aliases -------------------------------------------------------------
    add_alias("p1/RandomSampling/nq100", "shared/random_nq100", scale.paper1_params,
              "Paper 1's RandomSampling@100 row IS the shared baseline run.", "RandomSampling")
    add_alias(f"p1/{KMH}/nq100", "shared/kmh_nq100", kmh_params,
              "Paper 1's KMH@100 row IS the shared run (ADR 0009).")
    for method, tag, canon in (("rnhal", "RNHAL", "rnhal/6_3/stage_full"),
                               ("texhal", "TexHAL", "shared/texhal_full")):
        why = (f"{tag} 'full' (representation + reference hierarchy [600,200,100]) is ONE run "
               f"(stage_full.json == rep_full.json == the 6.2 reference row).")
        if method == "texhal":
            add_alias("texhal/6_3/stage_full", canon, (abl % method) + "stage_full.json", why)
        add_alias(f"{method}/6_1/rep_full", canon, (abl % method) + "rep_full.json", why)
        add_alias(f"{method}/6_2/ref", canon, (abl % method) + "stage_full.json", why)
        add_alias(f"{method}/6_3/stage_no_representation", "shared/kmh_nq100", kmh_params,
                  "w/o representation module = ImageNet ResNet-50 features + reference hierarchy = "
                  "the shared KMH@100 run (byte-identical config; ADR 0009).")

    q17_ref = "texhal/6_1/rep_q17"
    q17_why = ("TexHAL canonical Q=17 (ADR 0011): the reference hierarchy L=3 k=[600,200,100] with VCTex "
               "q=[17] IS the texhal/6_1/rep_q17 run (byte-identical config).")
    add_alias("texhal_q17/6_2/hier_ref", q17_ref, (abl % "texhal") + "rep_q17.json", q17_why)
    add_alias("texhal_q17/6_3/stage_full", q17_ref, (abl % "texhal") + "rep_q17.json", q17_why)
    add_alias("texhal_q17/6_3/stage_no_representation", "shared/kmh_nq100", kmh_params,
              "w/o representation module = ImageNet ResNet-50 features + reference hierarchy = "
              "the shared KMH@100 run (byte-identical config; ADR 0009).")

    for g in groups:
        if g["id"] in alias_notes:
            extra = "Also serves as: " + ", ".join(alias_notes[g["id"]]) + "."
            g["note"] = (g["note"] + " " + extra).strip()

    # --- tables --------------------------------------------------------------
    def row(label: str, run: str) -> dict[str, str]:
        return {"label": label, "run": run}

    def label_of(method: str) -> str:
        return KMH_LABEL if method == KMH else method

    paper1_tables = []
    for paper_nq in (10, 50, 100):
        rows = [row(label_of(m), f"p1/{m}/nq{paper_nq}") for m in PAPER1_METHODS]
        rows.append(row("Upper bound (full pool)", "p1/upper_bound"))
        paper1_tables.append({
            "id": f"paper1_nq{paper_nq}", "kind": "benchmark",
            "title": f"Paper 1 -- DAL benchmark, n_query={paper_nq}", "rows": rows,
        })

    hier_labels = dict(HIER_COMMON + HIER_NEW_RNHAL) | HIER_TRANSFER_LABELS

    def hier_rows(method: str, order: list[str], ref_id: str | None = None) -> list[dict[str, str]]:
        ref = ref_id or f"{method}/6_2/ref"
        return [row(REF_LABEL, ref) if key == "ref"
                else row(hier_labels[key], f"{method}/6_2/{key}") for key in order]

    def stage_tables(method: str, tag: str, suffix: str = "") -> list[dict]:
        return [
            {"id": f"6_2{suffix}", "kind": "ablation", "title": f"6.2 Hierarchy ablation{suffix_title(suffix)}",
             "rows": []},  # rows filled by the caller (hierarchy rows differ per variant)
            {"id": f"6_3{suffix}", "kind": "ablation",
             "title": f"6.3 Contribution of the two stages{suffix_title(suffix)}", "rows": [
                row(f"{tag} (full)", f"{method}/6_3/stage_full"),
                row("w/o representation module", f"{method}/6_3/stage_no_representation"),
                row("w/o hierarchical module", f"{method}/6_3/stage_no_hierarchy"),
            ]},
        ]

    def suffix_title(suffix: str) -> str:
        return " (history, VCTex Q=[5,17])" if suffix == "_q5_17" else ""

    def method_tables(method: str, tag: str, rep_rows: list[dict[str, str]], order: list[str],
                      suffix: str = "") -> list[dict]:
        t62, t63 = stage_tables(method, tag, suffix)
        t62["rows"] = hier_rows(method, order)
        return [{"id": "6_1", "kind": "ablation", "title": "6.1 Representation ablation", "rows": rep_rows},
                t62, t63]

    best = {"label": "Best paper-1 method @ n_query=100 (computed)",
            "best_of": {"candidates": [{"run": f"p1/{m}/nq100", "name": m} for m in PAPER1_METHODS],
                       "metric": "f1_score"}}
    random_row = row("RandomSampling @ n_query=100", "p1/RandomSampling/nq100")
    texhal_full = row("TexHAL (VCTex Q=17 + hierarchical)", "texhal/6_1/rep_q17")
    rnhal_full = row("RNHAL (SSRAE + hierarchical)", "rnhal/6_3/stage_full")

    rnhal_rep = [row("Full", "rnhal/6_1/rep_full")] + [
        row(label, f"rnhal/6_1/{cfg}") for cfg, label in RNHAL_61]
    texhal_rep = [row(label, f"texhal/6_1/{cfg}") for cfg, label in TEXHAL_61] + [
        row("Q=[5,17]", "texhal/6_1/rep_full")]

    # Paper 2 (ADR 0011): 6.1 unchanged; 6.2/6.3 from the texhal_q17 groups; the executed Q=[5,17]
    # 6.2/6.3 stay reachable as `6_2_q5_17` / `6_3_q5_17`.
    q17_6_2 = {"id": "6_2", "kind": "ablation", "title": "6.2 Hierarchy ablation",
               "rows": hier_rows("texhal_q17", HIER_ORDER_TEXHAL + HIER_TRANSFER_ORDER,
                                 "texhal_q17/6_2/hier_ref")}
    q17_6_3 = {"id": "6_3", "kind": "ablation", "title": "6.3 Contribution of the two stages", "rows": [
        row("TexHAL (full)", "texhal_q17/6_3/stage_full"),
        row("w/o representation module", "texhal_q17/6_3/stage_no_representation"),
        row("w/o hierarchical module", "texhal_q17/6_3/stage_no_hierarchy"),
    ]}
    legacy = method_tables("texhal", "TexHAL", texhal_rep, HIER_ORDER_TEXHAL, "_q5_17")
    paper2 = [legacy[0], q17_6_2, q17_6_3, legacy[1], legacy[2]] + [{
        "id": "comparison", "kind": "comparison",
        "title": "Paper 2 -- TexHAL vs. paper-1 methods, n_query=100",
        "rows": [texhal_full, best, random_row],
    }]
    paper3 = method_tables("rnhal", "RNHAL", rnhal_rep, HIER_ORDER_RNHAL + HIER_TRANSFER_ORDER) + [{
        "id": "comparison", "kind": "comparison",
        "title": "Paper 3 -- RNHAL vs. TexHAL vs. RandomSampling, n_query=100",
        "rows": [rnhal_full, texhal_full, random_row, best],
    }]
    tables = {
        "paper1": {"tables": paper1_tables},
        "paper2": {"tables": paper2},
        "paper3": {"tables": paper3},
    }

    used = derive_used_by(tables, {a["id"]: Alias(a["id"], a["canonical"], a["params_json"],
                                                  a["strategy_name"], a["reason"]) for a in aliases})
    for g in groups:
        g["used_by"] = used.get(g["id"], [])

    return {
        "name": scale.name,
        "description": "Single source of truth for the clean re-execution of everything papers 1-3 need on "
                       "DATA/daninhas_full_fixed (.specs/experiments/campaign.md, ADR 0008/0009/0016). Generated by "
                       "scripts/campaign/build_manifest.py -- do not edit by hand.",
        "dataset_name": "DANINHAS",
        "results_root": scale.results_root,
        **manifest_header(scale),
        "seeds": SEEDS,
        "primary_nq": primary,
        "groups": groups,
        "aliases": aliases,
        "tables": tables,
    }


def build_control_ki43() -> dict[str, Any]:
    """KI-43 control: RandomSampling nq100 re-run on ONE runtime image into a NEW results root.

    Same configuration as `shared/random_nq100` (paper-1 params, nq 100, NIL 100, NR 8, seeds 1-3) so
    its round-0 / final metrics can be compared with the campaign runs at the same seeds. No aliases
    and no tables (so `used_by` is empty and the manifest validates trivially). Since ADR 0016 it runs on
    DATA/daninhas_full_fixed (the params it shares with the campaign) into a root of its own.
    """
    root = "results/campaign_control_fixed"
    return {
        "name": "campaign_control_ki43",
        "description": "KI-43 control run (.specs/experiments/campaign.md, 'Control run for KI-43'). "
                       "Generated by scripts/campaign/build_manifest.py -- do not edit by hand.",
        "dataset_name": "DANINHAS",
        "results_root": root,
        "required_leaf_files": list(ROUND_HISTORY_FILES),
        "seeds": SEEDS,
        "primary_nq": FULL.nq_by_paper_nq[100],
        "groups": [{
            "id": "control/random_nq100_ki43", "part": "control",
            "params_json": FULL.paper1_params, "strategy_name": "RandomSampling",
            "n_query": FULL.nq_by_paper_nq[100], "n_init_labeled": FULL.n_init, "n_round": FULL.n_round,
            "seeds": list(SEEDS), "dir_results": f"{root}/random_nq100/", "used_by": [],
            "note": "KI-43 control: RandomSampling @ n_query=100 re-executed in ONE session on ONE runtime "
                    "image. Compare against results/campaign_fixed/shared/random_nq100/ at the same seeds "
                    "(round-0 and final weighted F1) to measure the cross-runtime deviation.",
        }],
        "aliases": [],
        "tables": {},
    }


def _stage2_rows(row: Any, label: Any, by_name: dict[str, Any]) -> list[dict[str, str]]:
    """Table rows of the stage-2 table: stage-1 context rows first, then the stage-2 points in grid order."""
    nq = sweep.STAGE2_NQ
    rows = [row(f"{label(by_name[n], nq)}", f"kmh/nq{nq}/{n}")
            for n in ("hier_L3_ref", "hier_L4_k800_600_200_100", "hier_L2_matched", "hier_L1_k100")]
    for name, h, knob in sweep.stage2_points():
        text = label(h, nq)
        if knob is not None:
            text = f"{text.split(' (')[0]} + {knob[1]}={knob[2]}"
        rows.append(row(text, f"kmh/nq{nq}/{name}"))
    return rows


def build_kmh_sweep(scale: Scale) -> dict[str, Any]:
    """Paper-1 KMH parameter sweep: stage 1 (hierarchy + knob axes at nq {10, 50, 100}) and stage 2 (nq100 only).

    One group per grid point (`kmh/nq{N}/<config>`, part `kmh_nq{N}`), each in its own results
    directory (every group shares strategy/NQ/NIL/NR, so a shared directory would make the leaf glob
    ambiguous). At nq100 the budget-matched L3 shape IS the reference: `kmh/nq100/hier_L3_matched` is an
    alias of `kmh/nq100/hier_L3_ref` (and `hier_L1_k{nq}` is `hier_L1_k100`, so it is not listed).
    Results go to a root of their own (`scale.sweep_results_root`); the reference is re-run inside it so
    every comparison of a sweep table is within one runtime image (KI-43) and one GPU part (ADR 0016).
    """
    root = scale.sweep_results_root
    params_dir = f"files_config/ablations/kmh_sweep/{scale.abl_subdir}"
    by_name = {h.name: h for h in sweep.HIERARCHIES}
    groups: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    def gid(paper_nq: int, name: str) -> str:
        return f"kmh/nq{paper_nq}/{name}"

    # Execution order (= PARTS order in dalmax/campaign.py): nq100 first, the smallest part and the most
    # informative one for papers 2 and 3, then nq50, then nq10.
    for paper_nq in sorted(sweep.BUDGETS, reverse=True):
        nq = scale.nq_by_paper_nq[paper_nq]
        part = f"kmh_nq{paper_nq}"
        for name, h, knob in sweep.grid(paper_nq):
            clusters = sweep.clusters_for(h, paper_nq)
            note = (f"KMH sweep @ n_query={paper_nq}: hierarchy k={sweep._fmt(clusters)}"
                    + (f", knob {knob[1]}={knob[2]!r}." if knob else " (hierarchy axis, other knobs default)."))
            groups.append({
                "id": gid(paper_nq, name), "part": part, "gpu_part": SWEEP_GPU_PART[paper_nq],
                "gpu_stage": STAGE_KMH_SWEEP, "params_json": f"{params_dir}nq{paper_nq}/{name}.json",
                "strategy_name": "RepresentationStrategy", "n_query": nq,
                "n_init_labeled": scale.n_init, "n_round": scale.n_round, "seeds": list(SEEDS),
                "dir_results": f"{root}/nq{paper_nq}/{name}/", "used_by": [], "note": note,
            })
        ref_id = gid(paper_nq, "hier_L3_ref")
        if paper_nq == sweep.REFERENCE_NQ:
            aliases.append({
                "id": gid(paper_nq, "hier_L3_matched"), "canonical": ref_id,
                "params_json": f"{params_dir}nq{paper_nq}/hier_L3_ref.json",
                "strategy_name": "RepresentationStrategy",
                "reason": "At n_query=100 the budget-matched L=3 hierarchy [6nq,2nq,nq] = [600,200,100] IS the "
                          "reference hierarchy: one run.",
            })
            for g in groups:
                if g["id"] == ref_id:
                    g["note"] += f" Also serves as: {gid(paper_nq, 'hier_L3_matched')}."

        def row(label: str, run: str) -> dict[str, str]:
            return {"label": label, "run": run}

        def label(h: sweep.Hierarchy, budget: int) -> str:
            return h.label.format(k=sweep._fmt(sweep.clusters_for(h, budget)))

        tables.append({
            "id": f"kmh_hier_nq{paper_nq}", "kind": "ablation",
            "title": f"KMH sweep -- hierarchy axis, n_query={paper_nq}",
            "rows": [row(label(h, paper_nq), gid(paper_nq, sweep.config_name(h, paper_nq)))
                     for h in sweep.hierarchies_for(paper_nq)],
        })
        knob_rows: list[dict[str, str]] = []
        for hname in sweep.KNOB_HIERARCHIES:
            h = by_name[hname]
            if sweep.is_alias_of_reference(h, paper_nq):
                continue
            tag = "reference" if h.kind == "reference" else "budget-matched"
            knob_rows.append(row(label(h, paper_nq), gid(paper_nq, hname)))
            knob_rows += [row(f"{tag} + {key}={value}", gid(paper_nq, f"{hname}__{suffix}"))
                          for suffix, key, value in sweep.KNOBS]
        tables.append({
            "id": f"kmh_knobs_nq{paper_nq}", "kind": "ablation",
            "title": f"KMH sweep -- knob axis, n_query={paper_nq}", "rows": knob_rows,
        })
        if paper_nq == sweep.STAGE2_NQ:
            tables.append({
                "id": "kmh_stage2_nq100", "kind": "ablation",
                "title": "KMH sweep -- stage 2 (nq100): deeper hierarchies, coarse level below budget, "
                         "resampling on L=4",
                "rows": _stage2_rows(row, label, by_name),
            })

    all_tables = {"paper1": {"tables": tables}}
    used = derive_used_by(all_tables, {a["id"]: Alias(a["id"], a["canonical"], a["params_json"],
                                                      a["strategy_name"], a["reason"]) for a in aliases})
    for g in groups:
        g["used_by"] = used.get(g["id"], [])
    return {
        "name": f"{scale.name}_kmh_sweep",
        "description": "Paper-1 KMH parameter sweep, stages 1 and 2 (.specs/experiments/campaign.md, 'KMH parameter "
                       "sweep'). Generated by scripts/campaign/build_manifest.py -- do not edit by hand.",
        "dataset_name": "DANINHAS",
        "results_root": root,
        **manifest_header(scale),
        "seeds": SEEDS,
        "primary_nq": scale.nq_by_paper_nq[100],
        "groups": groups,
        "aliases": aliases,
        "tables": all_tables,
    }


TARGETS = {
    "files_config/campaign/manifest.json": lambda: build(FULL),
    "files_config/campaign/manifest_micro.json": lambda: build(MICRO),
    "files_config/campaign/manifest_control_ki43.json": build_control_ki43,
    "files_config/campaign/manifest_kmh_sweep.json": lambda: build_kmh_sweep(FULL),
    "files_config/campaign/manifest_kmh_sweep_micro.json": lambda: build_kmh_sweep(MICRO),
}


def render(builder: Callable[[], dict[str, Any]]) -> str:
    return json.dumps(builder(), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed manifest differs")
    args = parser.parse_args()
    drift = False
    for rel, builder in TARGETS.items():
        path = REPO_ROOT / rel
        text = render(builder)
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"DRIFT: {rel}")
                drift = True
        else:
            path.write_text(text)
            print(f"wrote {rel}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
