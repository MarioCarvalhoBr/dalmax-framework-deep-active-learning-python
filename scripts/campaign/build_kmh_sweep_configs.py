"""Generate the KMH parameter-sweep params files under `files_config/ablations/kmh_sweep/`.

Stage 1 of the paper-1 KMH sweep (`.specs/experiments/campaign.md`, "KMH parameter sweep"): for each
budget `n_query` in {10, 50, 100} a hierarchy axis (different `n_clusters`) and a knob axis (one
non-default hierarchical-k-means knob at a time, applied to the reference and the budget-matched
3-level hierarchy). The grid is plain data at the top of this file so that a later stage can extend it:
stage 2 (nq100, 2026-11-08), the wide-top exploration `kmh_wide` (`WIDE_POINTS`, 2026-11-15, ADR 0018) and
the feature post-processing round `kmh_norm` (`NORM_VARIANTS`, 2026-11-15, ADR 0019: the reference hierarchy
with an `embedding.postprocess` block, `hier_L3_ref__norm_<variant>.json`) are appended that way, every
earlier file byte-identical.

Full-scale files go to `kmh_sweep/nq{N}/<config>.json`, the CPU smoke mirror to
`kmh_sweep/micro/nq{N}/<config>.json`. Committed artifacts; `tests/test_kmh_sweep_configs.py` fails
if they drift. Run from the repo root:

    poetry run python scripts/campaign/build_kmh_sweep_configs.py           # rewrite all
    poetry run python scripts/campaign/build_kmh_sweep_configs.py --check   # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = "files_config/ablations/kmh_sweep"

# --- the grid (data) -------------------------------------------------------------------------------

BUDGETS: tuple[int, ...] = (10, 50, 100)
REFERENCE_CLUSTERS: tuple[int, ...] = (600, 200, 100)
REFERENCE_SAMPLE_SIZES: tuple[int, ...] = (30, 15, 2)  # historical, kept as is (not the 0.05 rule)
SAMPLE_RATIO_PERCENT = 5  # sample_sizes[i] = max(2, round_half_up(0.05 * n_clusters[i])) for every other shape
MICRO_DIVISOR = 10  # micro mirror: clusters / 10 (min 2), same sample-size rule on the scaled clusters
MIN_SAMPLE = 2


@dataclass(frozen=True)
class Hierarchy:
    """One point of the hierarchy axis. `clusters(nq)` gives the per-level cluster counts."""

    name: str  # config name; `{nq}` is replaced by the budget
    label: str  # table label; `{k}` is replaced by the cluster list, e.g. "[60,10]"
    kind: str  # "fixed" | "matched" | "reference"
    base: tuple[int, ...] = ()  # fixed: the clusters; matched: multipliers of nq per level
    sample_factor: int = 1  # multiplies the rule-derived sample_sizes (capped at the cluster count)


# Order = row order of the `kmh_hier_nq*` tables.
HIERARCHIES: tuple[Hierarchy, ...] = (
    Hierarchy("hier_L1_k{nq}", "L=1, k={k} (= n_query)", "matched", (1,)),
    Hierarchy("hier_L1_k100", "L=1, k={k}", "fixed", (100,)),
    Hierarchy("hier_L1_k200", "L=1, k={k}", "fixed", (200,)),
    Hierarchy("hier_L1_k600", "L=1, k={k}", "fixed", (600,)),
    Hierarchy("hier_L2_k200_100", "L=2, k={k}", "fixed", (200, 100)),
    Hierarchy("hier_L2_matched", "L=2, k={k} (budget-matched)", "matched", (6, 1)),
    Hierarchy("hier_L3_ref", "L=3, k={k} (reference)", "reference", REFERENCE_CLUSTERS),
    Hierarchy("hier_L3_matched", "L=3, k={k} (budget-matched)", "matched", (6, 2, 1)),
    Hierarchy("hier_L4_k800_600_200_100", "L=4, k={k}", "fixed", (800, 600, 200, 100)),
)
# Knob axis: (suffix, hierarchy-block key, value). Applied to these hierarchies (nq100: reference only).
KNOBS: tuple[tuple[str, str, Any], ...] = (
    ("init_random", "init_method", "random"),
    ("leaf_closest", "leaf_sampling", "closest"),
    ("no_resampling", "n_resamples", 0),
    ("resample_random", "resample_strategy", "random"),
)
KNOB_HIERARCHIES: tuple[str, ...] = ("hier_L3_ref", "hier_L3_matched")
REFERENCE_NQ = 100  # at this budget the matched shapes coincide with the reference / the k100 file

# Stage 2 (2026-11-08, nq100 only; appended after the stage-1 files, stage-1 files stay byte-identical).
STAGE2_NQ = 100
STAGE2_HIERARCHIES: tuple[Hierarchy, ...] = (
    Hierarchy("hier_L5_k1000_800_600_200_100", "L=5, k={k}", "fixed", (1000, 800, 600, 200, 100)),
    Hierarchy("hier_L4_k1200_600_200_100", "L=4, k={k}", "fixed", (1200, 600, 200, 100)),
    Hierarchy("hier_L4_k800_600_200_50", "L=4, k={k}", "fixed", (800, 600, 200, 50)),
    Hierarchy("hier_L3_k800_600_100", "L=3, k={k}", "fixed", (800, 600, 100)),
    Hierarchy("hier_L1_k50", "L=1, k={k}", "fixed", (50,)),
    Hierarchy("hier_L1_k20", "L=1, k={k}", "fixed", (20,)),
)
# Stage-2 knob combinations: (hierarchy name, knob suffix); the hierarchy may be a stage-1 one.
STAGE2_KNOB_COMBOS: tuple[tuple[str, str], ...] = (
    ("hier_L4_k800_600_200_100", "no_resampling"),
    ("hier_L4_k800_600_200_100", "resample_random"),
)
# Resampling-intensity point: same clusters as hier_L4_k800_600_200_100, sample_sizes doubled.
STAGE2_SAMPLE_X2 = Hierarchy("hier_L4_k800_600_200_100__sample_x2", "L=4, k={k} + sample_sizes x2",
                             "fixed", (800, 600, 200, 100), sample_factor=2)
STAGE2_LATE_KNOB_COMBOS: tuple[tuple[str, str], ...] = (("hier_L2_matched", "no_resampling"),)

# Wide-top exploration `kmh_wide` (2026-11-15, ADR 0018; appended, every earlier file stays byte-identical):
# fixed shapes per budget, every knob at its default, sample sizes by the 0.05 rule. Direction (a): a wide
# first level with the last level equal to the query size (never run at nq10/nq50); direction (b): a wider
# first level (one level k=1200 / 2400, three levels [1200,400,nq]). Order = row order of `kmh_wide_nq*`.
WIDE_POINTS: dict[int, tuple[Hierarchy, ...]] = {
    100: (
        Hierarchy("hier_L1_k1200", "L=1, k={k}", "fixed", (1200,)),
        Hierarchy("hier_L1_k2400", "L=1, k={k}", "fixed", (2400,)),
        Hierarchy("hier_L3_k1200_400_100", "L=3, k={k}", "fixed", (1200, 400, 100)),
    ),
    50: (
        Hierarchy("hier_L3_k600_200_50", "L=3, k={k} (last level = n_query)", "fixed", (600, 200, 50)),
        Hierarchy("hier_L2_k600_50", "L=2, k={k} (last level = n_query)", "fixed", (600, 50)),
        Hierarchy("hier_L1_k1200", "L=1, k={k}", "fixed", (1200,)),
        Hierarchy("hier_L1_k2400", "L=1, k={k}", "fixed", (2400,)),
        Hierarchy("hier_L3_k1200_400_50", "L=3, k={k}", "fixed", (1200, 400, 50)),
    ),
    10: (
        Hierarchy("hier_L3_k600_200_10", "L=3, k={k} (last level = n_query)", "fixed", (600, 200, 10)),
        Hierarchy("hier_L2_k600_10", "L=2, k={k} (last level = n_query)", "fixed", (600, 10)),
        Hierarchy("hier_L1_k1200", "L=1, k={k}", "fixed", (1200,)),
        Hierarchy("hier_L1_k2400", "L=1, k={k}", "fixed", (2400,)),
        Hierarchy("hier_L3_k1200_400_10", "L=3, k={k}", "fixed", (1200, 400, 10)),
    ),
}

# Feature post-processing round `kmh_norm` (2026-11-15, ADR 0019; appended, every earlier file stays
# byte-identical): the REFERENCE hierarchy (fixed beforehand, so the round depends on no choice made on the test
# set), every knob at its default, at every budget, with a non-identity `embedding.postprocess` block. Entries:
# (variant = suffix of the config name and name of the results folder, the params block, the table label).
# Order = row order of the `kmh_norm_nq*` tables, after the executed raw-feature reference row.
NORM_HIERARCHY = "hier_L3_ref"
NORM_VARIANTS: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("l2", {"method": "l2"}, "L2-normalised features"),
    ("zscore", {"method": "zscore"}, "per-dimension z-score (initial-pool mean and std)"),
    ("pca64_l2", {"method": "pca_l2", "pca_dim": 64}, "PCA to 64 dimensions (initial pool) + L2"),
)
# The explicit identity: the params file of each table's alias row, the executed raw-feature reference
# `kmh/nq{N}/hier_L3_ref` (the manifest checks that its effective signature equals the reference's).
NORM_IDENTITY: tuple[str, dict[str, Any], str] = ("none", {"method": "none"}, "raw features (executed reference)")

# --- derivation ------------------------------------------------------------------------------------


def config_name(h: Hierarchy, nq: int) -> str:
    return h.name.format(nq=nq)


def clusters_for(h: Hierarchy, nq: int) -> tuple[int, ...]:
    return tuple(m * nq for m in h.base) if h.kind == "matched" else h.base


def is_alias_of_reference(h: Hierarchy, nq: int) -> bool:
    """At nq100 the budget-matched shapes ARE existing files: L3 = the reference, L1 = hier_L1_k100."""
    return nq == REFERENCE_NQ and h.kind == "matched" and h.name != "hier_L2_matched"


def hierarchies_for(nq: int) -> list[Hierarchy]:
    """Hierarchies that are real runs at `nq` (the nq100 L1_k{nq} duplicate of hier_L1_k100 is dropped)."""
    return [h for h in HIERARCHIES if not (nq == REFERENCE_NQ and h.name == "hier_L1_k{nq}")]


def sample_sizes_for(clusters: tuple[int, ...], *, reference: bool = False, factor: int = 1) -> tuple[int, ...]:
    if reference:
        return REFERENCE_SAMPLE_SIZES
    # round_half_up(0.05 * k) in integer arithmetic, floor MIN_SAMPLE, times factor, never above k
    return tuple(min(k, factor * max(MIN_SAMPLE, (k * SAMPLE_RATIO_PERCENT + 50) // 100)) for k in clusters)


def scale_clusters_micro(clusters: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(2, k // MICRO_DIVISOR) for k in clusters)


def _fmt(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def stage2_points() -> list[tuple[str, Hierarchy, tuple[str, str, Any] | None]]:
    """Stage-2 `(config name, hierarchy, knob)` in table-row order; nq100 only."""
    by_name = {h.name: h for h in HIERARCHIES}
    knobs = {k[0]: k for k in KNOBS}
    points: list[tuple[str, Hierarchy, tuple[str, str, Any] | None]] = [
        (h.name, h, None) for h in STAGE2_HIERARCHIES]
    for hname, suffix in STAGE2_KNOB_COMBOS:
        points.append((f"{hname}__{suffix}", by_name[hname], knobs[suffix]))
    points.append((STAGE2_SAMPLE_X2.name, STAGE2_SAMPLE_X2, None))
    for hname, suffix in STAGE2_LATE_KNOB_COMBOS:
        points.append((f"{hname}__{suffix}", by_name[hname], knobs[suffix]))
    return points


def is_stage2(h: Hierarchy, knob: tuple[str, str, Any] | None, nq: int) -> bool:
    name = h.name if knob is None else f"{h.name}__{knob[0]}"
    return nq == STAGE2_NQ and name in {n for n, _, _ in stage2_points()}


def build_config(nq: int, h: Hierarchy, knob: tuple[str, str, Any] | None, *, micro: bool,
                 wide: bool = False, postprocess: dict[str, Any] | None = None) -> dict[str, Any]:
    """The params JSON content of one grid point (`knob` None = the hierarchy-axis config; `wide` = a
    `kmh_wide` point, ADR 0018; `postprocess` = the `embedding.postprocess` block of a `kmh_norm` point,
    ADR 0019, None = no such key, as in every earlier file)."""
    full_clusters = clusters_for(h, nq)
    reference = h.kind == "reference"
    full_samples = sample_sizes_for(full_clusters, reference=reference, factor=h.sample_factor)
    if micro:
        clusters = scale_clusters_micro(full_clusters)
        samples = sample_sizes_for(clusters, factor=h.sample_factor)
    else:
        clusters, samples = full_clusters, full_samples
    hierarchy: dict[str, Any] = {
        "n_clusters": list(clusters),
        "n_levels": len(clusters),
        "sample_sizes": list(samples),
    }
    what = f"hierarchy k={_fmt(clusters)} sample_sizes={_fmt(samples)}"
    if knob is not None:
        suffix, key, value = knob
        hierarchy[key] = value
        what += f", knob {key}={value!r} (default elsewhere)"
    stage = 2 if is_stage2(h, knob, nq) else 1
    scope = "Micro (CPU smoke) mirror of" if micro else f"KMH sweep stage {stage}:"
    comment = (
        f"{scope} ImageNet ResNet-50 features + hierarchical k-means at n_query={nq}, {what}; otherwise "
        "identical to params_kmh.json. See .specs/experiments/campaign.md, 'KMH parameter sweep'."
    )
    if stage == 2 and not micro:
        comment = (f"KMH sweep stage 2 (nq100 only, 2026-11-08): ImageNet ResNet-50 features + hierarchical "
                   f"k-means at n_query={nq}, {what}; otherwise identical to params_kmh.json. See "
                   ".specs/experiments/campaign.md, 'KMH parameter sweep'.")
    if wide and not micro:
        comment = (f"KMH wide-top exploration (kmh_wide, 2026-11-15, ADR 0018): ImageNet ResNet-50 features + "
                   f"hierarchical k-means at n_query={nq}, {what}; otherwise identical to params_kmh.json. See "
                   ".specs/experiments/campaign.md, 'KMH wide-top exploration'.")
    embedding: dict[str, Any] = {"extractor": "resnet_imagenet", "q": None, "variant": "full"}
    if postprocess is not None:
        embedding["postprocess"] = postprocess
        what += ", embedding.postprocess " + ", ".join(f"{key}={value}" for key, value in postprocess.items())
        if not micro:
            identity = postprocess == NORM_IDENTITY[1]
            comment = (f"KMH feature post-processing (kmh_norm, 2026-11-15, ADR 0019): ImageNet ResNet-50 features + "
                       f"hierarchical k-means at n_query={nq}, {what}; otherwise identical to {NORM_HIERARCHY}.json "
                       "of the same budget. "
                       + ("The explicit identity: the same computation as that file (the manifest's alias of the "
                          "executed reference row). " if identity else "")
                       + "See .specs/experiments/campaign.md, 'KMH feature post-processing'.")
    if micro:
        comment = (f"Micro (CPU smoke) mirror (clusters / {MICRO_DIVISOR}, min 2) of the full-scale "
                   f"config of the same name: n_query={nq}, {what}.")
    return {
        "_comment": comment,
        "DANINHAS": {
            "data_dir": "DATA/daninhas_micro/" if micro else "DATA/daninhas_full_fixed/",
            "n_epoch": 1 if micro else 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 16 if micro else 256, "num_workers": 0 if micro else 4},
            "test_args": {"batch_size": 16 if micro else 256, "num_workers": 0 if micro else 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
            "embedding": embedding,
            "selection": {"method": "hierarchical", "hierarchy": hierarchy},
        },
    }


def grid(nq: int) -> list[tuple[str, Hierarchy, tuple[str, str, Any] | None]]:
    """`(config name, hierarchy, knob or None)` for every file at budget `nq` (hierarchy axis then knob axis)."""
    items: list[tuple[str, Hierarchy, tuple[str, str, Any] | None]] = []
    for h in hierarchies_for(nq):
        if not is_alias_of_reference(h, nq):
            items.append((config_name(h, nq), h, None))
    by_name = {h.name: h for h in HIERARCHIES}
    for hname in KNOB_HIERARCHIES:
        h = by_name[hname]
        if is_alias_of_reference(h, nq):
            continue
        for knob in KNOBS:
            items.append((f"{config_name(h, nq)}__{knob[0]}", h, knob))
    if nq == STAGE2_NQ:
        items += stage2_points()
    return items


def wide_points(nq: int) -> list[tuple[str, Hierarchy]]:
    """`(config name, hierarchy)` of the `kmh_wide` points at budget `nq` (ADR 0018), in table-row order."""
    return [(config_name(h, nq), h) for h in WIDE_POINTS.get(nq, ())]


def norm_config_name(variant: str) -> str:
    """Config name of a `kmh_norm` point: the reference hierarchy plus the post-processing variant."""
    return f"{NORM_HIERARCHY}__norm_{variant}"


def norm_points() -> list[tuple[str, str, dict[str, Any], str]]:
    """`(config name, variant, postprocess block, table label)` of the `kmh_norm` params files (ADR 0019), the
    explicit identity first, then the variants in table-row order; the same at every budget."""
    return [(norm_config_name(v), v, block, label) for v, block, label in (NORM_IDENTITY, *NORM_VARIANTS)]


def expected_files() -> dict[str, str]:
    """`{repo-relative path: file text}` of every generated params file."""
    files: dict[str, str] = {}
    reference = {h.name: h for h in HIERARCHIES}[NORM_HIERARCHY]
    for micro in (False, True):
        sub = f"{OUT_DIR}/micro" if micro else OUT_DIR
        for nq in BUDGETS:
            for name, h, knob in grid(nq):
                files[f"{sub}/nq{nq}/{name}.json"] = json.dumps(build_config(nq, h, knob, micro=micro), indent=4) + "\n"
            for name, h in wide_points(nq):
                rel = f"{sub}/nq{nq}/{name}.json"
                if rel in files:
                    raise ValueError(f"kmh_wide point {rel} collides with a sweep file")
                files[rel] = json.dumps(build_config(nq, h, None, micro=micro, wide=True), indent=4) + "\n"
            for name, _, block, _ in norm_points():
                rel = f"{sub}/nq{nq}/{name}.json"
                if rel in files:
                    raise ValueError(f"kmh_norm point {rel} collides with a sweep file")
                files[rel] = json.dumps(build_config(nq, reference, None, micro=micro, postprocess=block),
                                        indent=4) + "\n"
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed file differs or is missing")
    args = parser.parse_args()
    drift = False
    for rel, text in expected_files().items():
        path = REPO_ROOT / rel
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"DRIFT: {rel}")
                drift = True
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if not args.check:
        print(f"wrote {len(expected_files())} files under {OUT_DIR}/")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
