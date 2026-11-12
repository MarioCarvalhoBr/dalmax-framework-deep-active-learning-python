"""Generate the KMH parameter-sweep params files under `files_config/ablations/kmh_sweep/`.

Stage 1 of the paper-1 KMH sweep (`.specs/experiments/campaign.md`, "KMH parameter sweep"): for each
budget `n_query` in {10, 50, 100} a hierarchy axis (different `n_clusters`) and a knob axis (one
non-default hierarchical-k-means knob at a time, applied to the reference and the budget-matched
3-level hierarchy). The grid is plain data at the top of this file so that a later stage can extend it.

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


def build_config(nq: int, h: Hierarchy, knob: tuple[str, str, Any] | None, *, micro: bool) -> dict[str, Any]:
    """The params JSON content of one grid point (`knob` None = the hierarchy-axis config)."""
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
            "embedding": {"extractor": "resnet_imagenet", "q": None, "variant": "full"},
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


def expected_files() -> dict[str, str]:
    """`{repo-relative path: file text}` of every generated params file."""
    files: dict[str, str] = {}
    for micro in (False, True):
        sub = f"{OUT_DIR}/micro" if micro else OUT_DIR
        for nq in BUDGETS:
            for name, h, knob in grid(nq):
                files[f"{sub}/nq{nq}/{name}.json"] = json.dumps(build_config(nq, h, knob, micro=micro), indent=4) + "\n"
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
