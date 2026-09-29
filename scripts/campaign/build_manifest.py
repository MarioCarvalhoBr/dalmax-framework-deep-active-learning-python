"""Generate `files_config/campaign/manifest.json` and `manifest_micro.json`.

The manifests are committed artifacts (single source of truth, see
`.specs/experiments/campaign-a100.md`); this script is how they are produced,
and `tests/test_campaign.py` fails if the committed files drift from what it
emits. Run from the repo root:

    poetry run python scripts/campaign/build_manifest.py            # rewrite both
    poetry run python scripts/campaign/build_manifest.py --check    # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dalmax.campaign import Alias, derive_used_by  # noqa: E402

SEEDS = [1, 2, 3]
# Paper 1: the 12 classical strategies (no RNHAL/TexHAL -- those belong to papers 3/2).
CLASSICAL_STRATEGIES = [
    "RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling",
    "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout",
    "KMeansSampling", "KCenterGreedy", "BALDDropout", "AdversarialBIM",
    "AdversarialDeepFool",
]
# KMH = plain hierarchical k-means on ImageNet ResNet-50 features (no SSRAE/VCTex);
# interpretation CONFIRMED by the user 2026-09-29 (see .specs/experiments/campaign-a100.md).
KMH = "KMH"
KMH_LABEL = "KMH (hierarchical k-means, ImageNet ResNet-50 features)"
# Paper-1 methods eligible for "best paper-1 method @ nq=100": 12 classical + KMH.
PAPER1_METHODS = [*CLASSICAL_STRATEGIES, KMH]


@dataclass(frozen=True)
class Scale:
    """Everything that differs between the full-scale and the micro manifest."""

    name: str
    results_root: str
    paper1_params: str
    kmh_params: str
    abl_subdir: str  # "" or "micro/"
    nq_by_paper_nq: dict[int, int]  # paper nq -> actual --n_query
    n_init: int
    n_round: int
    upper_bound_n_init: int


FULL = Scale(
    "campaign_a100", "results/campaign_a100", "files_config/campaign/params_paper1.json",
    "files_config/campaign/params_kmh.json", "",
    {10: 10, 50: 50, 100: 100}, 100, 8, 8086,
)
MICRO = Scale(
    "campaign_micro", "results/smoke_campaign", "files_config/campaign/params_paper1_micro.json",
    "files_config/campaign/params_kmh_micro.json", "micro/",
    {10: 3, 50: 4, 100: 5}, 10, 1, 806,
)

# (id suffix, params basename, table label) -- ablation rows per study, per method.
RNHAL_61 = [("rep_spatial", "Spatial-only"), ("rep_spectral", "Spectral-only")]
TEXHAL_61 = [("rep_q5", "Q=5"), ("rep_q13", "Q=13"), ("rep_q17", "Q=17")]
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
                  dir_results: str, note: str = "") -> None:
        groups.append({
            "id": gid, "part": part, "params_json": params, "strategy_name": strategy,
            "n_query": nq, "n_init_labeled": nil, "n_round": nr, "seeds": list(SEEDS),
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

    # --- paper 1 (12 classical + KMH) x nq {100, 50, 10} ---------------------
    # First job of the whole campaign: the cheapest run (a quick GPU/environment sanity check).
    add_group("shared/random_nq100", "paper1", scale.paper1_params, "RandomSampling", primary,
              scale.n_init, scale.n_round, f"{root}/shared/random_nq100/", shared_note)
    for paper_nq in (100, 50, 10):
        nq = scale.nq_by_paper_nq[paper_nq]
        for strat in CLASSICAL_STRATEGIES:
            if paper_nq == 100 and strat == "RandomSampling":
                continue  # shared/random_nq100 (below): baseline of papers 1, 2 and 3
            add_group(f"p1/{strat}/nq{paper_nq}", "paper1", scale.paper1_params, strat, nq,
                      scale.n_init, scale.n_round, f"{root}/paper1/nq{paper_nq}/")
        if paper_nq == 100:
            add_group("shared/kmh_nq100", "paper1", kmh_params, "RepresentationStrategy", nq,
                      scale.n_init, scale.n_round, f"{root}/shared/kmh_nq100/",
                      "KMH @ n_query=100 = hierarchical k-means on ImageNet ResNet-50 features (reference "
                      "hierarchy [600,200,100]) = the 'w/o representation module' row of papers 2 and 3. "
                      + shared_note)
        else:
            add_group(f"p1/{KMH}/nq{paper_nq}", "paper1", kmh_params, "RepresentationStrategy", nq,
                      scale.n_init, scale.n_round, f"{root}/paper1/kmh/nq{paper_nq}/",
                      "KMH: hierarchical k-means selection on ImageNet ResNet-50 features (reference "
                      "hierarchy [600,200,100]), no SSRAE/VCTex.")
    add_group("p1/upper_bound", "upper_bound", scale.paper1_params, "FullSupervised", primary,
              scale.upper_bound_n_init, 0, f"{root}/paper1/upper_bound/",
              "Upper bound (not active learning): train once on the ENTIRE pool "
              "(n_init_labeled == pool size, n_round 0), evaluate on the fixed test set.")

    # --- rnhal / texhal ablation groups (full rows are ONE canonical run) -----
    def ablation_groups(method: str, study_rows: dict[str, list[str]]) -> None:
        for study, cfgs in study_rows.items():
            for cfg in cfgs:
                add_group(f"{method}/{study}/{cfg}", method, (abl % method) + f"{cfg}.json",
                          "RepresentationStrategy", primary, scale.n_init, scale.n_round,
                          f"{root}/{method}/{study}/{cfg}/")

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
    # TexHAL 'full' is consumed by paper 2 (6.1/6.2/6.3 + comparison) AND paper 3 (comparison).
    add_group("shared/texhal_full", "texhal", (abl % "texhal") + "stage_full.json",
              "RepresentationStrategy", primary, scale.n_init, scale.n_round,
              f"{root}/shared/texhal_full/",
              "TexHAL 'full' (VCTex q=[5,17] + reference hierarchy). " + shared_note)

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

    hier_labels = dict(HIER_COMMON + HIER_NEW_RNHAL)

    def hier_rows(method: str, order: list[str]) -> list[dict[str, str]]:
        return [row(REF_LABEL, f"{method}/6_2/ref") if key == "ref"
                else row(hier_labels[key], f"{method}/6_2/{key}") for key in order]

    def method_tables(method: str, tag: str, rep_rows: list[dict[str, str]], order: list[str]) -> list[dict]:
        return [
            {"id": "6_1", "kind": "ablation", "title": "6.1 Representation ablation", "rows": rep_rows},
            {"id": "6_2", "kind": "ablation", "title": "6.2 Hierarchy ablation",
             "rows": hier_rows(method, order)},
            {"id": "6_3", "kind": "ablation", "title": "6.3 Contribution of the two stages", "rows": [
                row(f"{tag} (full)", f"{method}/6_3/stage_full"),
                row("w/o representation module", f"{method}/6_3/stage_no_representation"),
                row("w/o hierarchical module", f"{method}/6_3/stage_no_hierarchy"),
            ]},
        ]

    best = {"label": "Best paper-1 method @ n_query=100 (computed)",
            "best_of": {"candidates": [{"run": f"p1/{m}/nq100", "name": m} for m in PAPER1_METHODS],
                       "metric": "f1_score"}}
    random_row = row("RandomSampling @ n_query=100", "p1/RandomSampling/nq100")
    texhal_full = row("TexHAL (VCTex + hierarchical)", "shared/texhal_full")
    rnhal_full = row("RNHAL (SSRAE + hierarchical)", "rnhal/6_3/stage_full")

    rnhal_rep = [row("Full", "rnhal/6_1/rep_full")] + [
        row(label, f"rnhal/6_1/{cfg}") for cfg, label in RNHAL_61]
    texhal_rep = [row(label, f"texhal/6_1/{cfg}") for cfg, label in TEXHAL_61] + [
        row("Q=[5,17] (full)", "texhal/6_1/rep_full")]

    paper2 = method_tables("texhal", "TexHAL", texhal_rep, HIER_ORDER_TEXHAL) + [{
        "id": "comparison", "kind": "comparison",
        "title": "Paper 2 -- TexHAL vs. paper-1 methods, n_query=100",
        "rows": [texhal_full, best, random_row],
    }]
    paper3 = method_tables("rnhal", "RNHAL", rnhal_rep, HIER_ORDER_RNHAL) + [{
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
        "description": "Single source of truth for the one-shot re-execution of everything papers 1-3 need "
                       "(.specs/experiments/campaign-a100.md, ADR 0008/0009). Generated by "
                       "scripts/campaign/build_manifest.py -- do not edit by hand.",
        "dataset_name": "DANINHAS",
        "results_root": scale.results_root,
        "seeds": SEEDS,
        "primary_nq": primary,
        "groups": groups,
        "aliases": aliases,
        "tables": tables,
    }


TARGETS = {
    "files_config/campaign/manifest.json": FULL,
    "files_config/campaign/manifest_micro.json": MICRO,
}


def render(scale: Scale) -> str:
    return json.dumps(build(scale), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed manifest differs")
    args = parser.parse_args()
    drift = False
    for rel, scale in TARGETS.items():
        path = REPO_ROOT / rel
        text = render(scale)
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
