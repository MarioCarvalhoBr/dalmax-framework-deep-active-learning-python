"""Build `files_config/campaign/gpu_time_estimates.json`: estimated GPU seconds per run of every group.

The table balances the two GPU parts of the campaign on `DATA/daninhas_full_fixed` (ADR 0016) and is
what `python -m dalmax.campaign list` sums into estimated hours per part and per GPU part. It is
committed DATA; this script documents how it was obtained and can rebuild it on a machine that holds
the trees of the first execution (read only, never written):

- **measured** groups: the same group id was executed in the first campaign (dataset with duplicates,
  KI-46) under `results/campaign/` or `results/campaign_kmh/`. Per run, the wall time of the trainer
  subprocess (`DONE elapsed=<s>` in that tree's `campaign.log`), or, for the runs that log does not
  hold (merged from another session), the run's own `Total time: <s> seconds` line of
  `log-dalmax.log` plus the median subprocess overhead measured on the runs that have both. The
  group's value is the mean over its seeds, rounded to whole seconds.
- **estimate** groups: the eight DeepAL+ strategies of ADR 0015 and the WAAL control, never executed on
  the GPU. Rules
  (author's brief of 2026-11-11, not measurements): `LossPredictionLoss`, `BadgeSampling` and
  `KCenterGreedyPCA` cost about a plain uncertainty strategy (the mean of `LeastConfidence`,
  `MarginSampling` and `EntropySampling` at the same budget); `VarRatio` and `MeanSTD` about an
  MC-dropout strategy (mean of the four at the same budget); `CEALSampling` 3 times a plain one
  (revised 2026-11-12 from 1.25: since the ADR 0015 follow-up CEAL trains on the labeled set plus the
  pseudo-labelled samples, up to about a quarter of the pool, at every round); `WAAL` about 45 minutes and
  `VAAL` (VAE images 32 x 32) about 50 minutes per run (FLOP estimates); the WAAL control
  `WAALRandomQuery` (2026-11-12) equal to `WAAL`.
- **estimate by twin** groups (ADR 0017, 2026-11-15): the `texhal_q13` part, never run in the first
  execution, takes the measured value of the same group id under `texhal_q17` (`TWIN_PREFIXES`): the
  same hierarchy and selection with a single VCTex RAE whose hidden size differs (13 instead of 17), a
  negligible part of a run.
- **estimate by fit** groups (ADR 0018, 2026-11-15): the KMH wide-top exploration `kmh_wide`, never run
  in the first execution (`FIT_PREFIXES`). Rule: at the same query size and the same number of levels,
  the least-squares line of the measured seconds per run against the total number of clusters
  `sum(n_clusters)`, over the measured KMH-sweep groups with every knob at its default (no `__` suffix
  in the id), evaluated at the group's own total. The cost of the selection grows linearly with the
  number of clusters: every Lloyd iteration of the vendored k-means recomputes the centroids in a Python
  loop over the clusters (`kmeans_gpu.compute_centroids`), at every level and resampling step; the
  measured groups follow such a line closely (at nq100, L=1 k 20..600 and the L=3..5 shapes up to a
  total of 2,700 clusters). The one-level k=1200/2400 points extrapolate beyond the measured one-level
  range (k <= 600): treat them as the least certain entries.
- **estimate by reference** groups (ADR 0019, 2026-11-15): the KMH feature post-processing round `kmh_norm`,
  never run in the first execution (`REFERENCE_PREFIXES`), takes the measured value of the reference
  hierarchy at the same query size (`kmh/nq{N}/hier_L3_ref`): the same hierarchy, selection and training,
  plus a post-processing of the frozen features once per run (negligible for `l2`/`zscore`; for `pca_l2` a full
  SVD of the 7,130 x 2,048 initial pool, measured 105 s on a loaded local CPU, i.e. a few percent of a run).
  `pca_l2` clusters 64-d instead of 2,048-d vectors and may well be faster overall; no speed-up is assumed.
- **estimate by budget scaling** groups (ADR 0020, 2026-11-16): the part `descriptors_nq` (RNHAL and TexHAL at
  n_query 50 and 10), never run in the first execution (`SCALED_PREFIXES`), takes the measured value of the
  method's executed n_query=100 run times the ratio of the measured paper-1 KMH runs at the two query sizes
  (`p1/KMH/nq{N}` / `shared/kmh_nq100`, `KMH_BY_BUDGET`): KMH is the same reference hierarchy and training on
  other features, so the ratio carries the cost of the query size (fewer images labeled, smaller training sets
  in the later rounds) and nothing else. The ratios are close to one (0.97 at nq50, 0.94 at nq10): the run is
  dominated by the selection over the whole pool and the training, not by the batch size. The first-execution
  runs this scales may or may not have computed the descriptor cache themselves (about 1.5 minutes per seed
  for SSRAE, under a minute for VCTex); either way a few percent of a run.
- **estimate by counterpart** groups (ADR 0020, 2026-11-16): the part `seeds_ext` (the eight headline methods at
  n_query 100 on the new seeds 4-6), never run in the first execution (`SEEDS_EXT_COUNTERPARTS`), takes the entry
  of its seeds-1-3 counterpart, which is the same computation on other seeds: the measured value where the
  counterpart is measured, the counterpart's own estimate otherwise (`p1/WAAL/nq100`, 45 minutes;
  `p1/BadgeSampling/nq100`, a plain uncertainty strategy). The descriptor
  caches of seeds 4-6 exist nowhere, so the first run of each extractor and seed computes them (SSRAE, VCTex Q=13,
  ImageNet ResNet-50; a few minutes per seed in all), not added to the estimate.
- **estimate by clean-campaign counterpart** groups (ADR 0021, 2026-11-16): the part `kmh_confirm` (KMH with
  l2-normalised features, the KMH reference, RandomSampling and WAAL on the new seeds 4-6), never run anywhere
  (`KMH_CONFIRM_COUNTERPARTS`), takes the wall time of its seeds-1-3 counterpart MEASURED ON THE CLEAN CAMPAIGN (the
  mean of the three `DONE elapsed=<s>` lines of the counterpart in the clean campaign's logs, written as constants
  because those logs live under `results/` on the author's machine only): the same computation on other seeds. The
  first-execution values of this table are not used for it on purpose: the clean campaign measured WAAL at about
  3,480 s (nq100) to 3,650 s (nq10) per run, against the 2,700 s FLOP estimate. The ImageNet feature cache of seeds
  4-6 does not exist on account A's Drive, so the first KMH run of each seed computes it (a few minutes, shared by
  the three query sizes because its key ignores n_query), not added to the estimate.

Caveats: the first execution ran on the dataset with duplicates (8,086 pool / 2,106 test images; the
new one has 7,230 / 1,008), on the GPU class of that execution, and `KMeansSampling` /
`KCenterGreedy` / the MC-dropout strategies were measured with the pre-ADR-0014 code. The numbers are
for balancing and budgeting, not a performance claim. No GPU model is named here or in the table.

Run from the repo root:

    poetry run python scripts/campaign/build_gpu_time_estimates.py           # rewrite the table
    poetry run python scripts/campaign/build_gpu_time_estimates.py --check   # exit 1 on drift
                                                                             # (skipped without the old trees)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dalmax.campaign import RunGroup, load_manifest  # noqa: E402

OUT = "files_config/campaign/gpu_time_estimates.json"
MANIFESTS = ("files_config/campaign/manifest.json", "files_config/campaign/manifest_kmh_sweep.json")
# New results root (ADR 0016) -> the root of the first execution that measured the same group ids.
MEASURED_ROOTS: dict[str, str] = {
    "results/campaign_fixed": "results/campaign",
    "results/campaign_kmh_fixed": "results/campaign_kmh",
}
DONE_RE = re.compile(r"\] (\S+) seed=(\d+) DONE elapsed=(\d+)s")
TOTAL_TIME_RE = re.compile(r"Total time: ([0-9.]+) seconds")

PLAIN = ("LeastConfidence", "MarginSampling", "EntropySampling")
MC_DROPOUT = ("LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout", "BALDDropout")
WAAL_SECONDS = 45 * 60.0
# strategy -> (reference family, multiplier, fixed seconds or None, rule text)
ESTIMATE_RULES: dict[str, tuple[tuple[str, ...], float, float | None, str]] = {
    "LossPredictionLoss": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "BadgeSampling": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "KCenterGreedyPCA": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "CEALSampling": (PLAIN, 3.0, None, "3 x a plain uncertainty strategy at the same budget (trains on labeled + "
                                      "pseudo-labelled samples, up to about a quarter of the pool)"),
    "VarRatio": (MC_DROPOUT, 1.0, None, "about an MC-dropout strategy at the same budget"),
    "MeanSTD": (MC_DROPOUT, 1.0, None, "about an MC-dropout strategy at the same budget"),
    "WAAL": ((), 1.0, WAAL_SECONDS, "about 45 minutes per run (FLOP estimate)"),
    # The WAAL control (ADR 0015 amendment of 2026-11-12): WAAL's training, a random query; the query is a
    # negligible part of a WAAL run, so the same estimate.
    "WAALRandomQuery": ((), 1.0, WAAL_SECONDS, "equal to WAAL (same training; the random query is negligible)"),
    "VAAL": ((), 1.0, 50 * 60.0, "about 50 minutes per run at VAE image size 32 (FLOP estimate)"),
}


# Id prefix of a never-run part -> id prefix of the measured part it is estimated from (same suffix).
TWIN_PREFIXES: dict[str, tuple[str, str]] = {
    "texhal_q13/": ("texhal_q17/", "equal to the measured texhal_q17 twin (same hierarchy and selection, single "
                                   "VCTex RAE with Q=13 instead of Q=17)"),
}


def twin_estimate(group: RunGroup, measured: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Entry of a never-run group estimated from its measured twin (`TWIN_PREFIXES`), or None."""
    for prefix, (twin_prefix, rule) in TWIN_PREFIXES.items():
        if group.id.startswith(prefix):
            twin = twin_prefix + group.id[len(prefix):]
            return {"seconds_per_run": measured[twin]["seconds_per_run"], "basis": "estimate", "rule": rule}
    return None


# Id prefix of a never-run part estimated by the per-(budget, levels) linear fit on the measured sweep groups.
FIT_PREFIXES: dict[str, str] = {
    "kmh_wide/": ("linear in the total number of clusters: least-squares line of the measured KMH-sweep "
                  "groups (default knobs) of the same query size and number of levels, at sum(n_clusters)={total}"),
}
SWEEP_ID_PREFIX = "kmh/"

# Id prefix of a never-run part -> (id of the measured group it is estimated from, `{nq}` = the group's budget
# folder, e.g. "nq50"; rule text).
REFERENCE_PREFIXES: dict[str, tuple[str, str]] = {
    "kmh_norm/": ("kmh/{nq}/hier_L3_ref", "equal to the measured reference hierarchy at the same query size "
                                          "(kmh/{nq}/hier_L3_ref): same hierarchy, selection and training plus a "
                                          "post-processing of the frozen features; no speed-up assumed for pca_l2"),
}


# Id prefix of a never-run part -> (id of the measured n_query=100 run of the same configuration, rule text).
SCALED_PREFIXES: dict[str, tuple[str, str]] = {
    "descriptors_nq/rnhal/": ("rnhal/6_3/stage_full", "the measured rnhal/6_3/stage_full (n_query=100) x the "
                                                      "measured KMH ratio {kmh} / shared/kmh_nq100"),
    "descriptors_nq/texhal_q13/": ("texhal/6_1/rep_q13", "the measured texhal/6_1/rep_q13 (n_query=100) x the "
                                                         "measured KMH ratio {kmh} / shared/kmh_nq100"),
}
# Group of the seeds_ext part (seeds 4-6) -> its seeds-1-3 counterpart (the same computation).
SEEDS_EXT_COUNTERPARTS: dict[str, str] = {
    "seeds_ext/RandomSampling": "shared/random_nq100",
    "seeds_ext/BALDDropout": "p1/BALDDropout/nq100",
    "seeds_ext/BadgeSampling": "p1/BadgeSampling/nq100",
    "seeds_ext/KMH": "shared/kmh_nq100",
    "seeds_ext/upper_bound": "p1/upper_bound",
    "seeds_ext/rnhal": "rnhal/6_3/stage_full",
    "seeds_ext/texhal_q13": "texhal/6_1/rep_q13",
    "seeds_ext/WAAL": "p1/WAAL/nq100",
}


def counterpart_estimate(group: RunGroup, entries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Entry of a seeds_ext group: its counterpart's seconds (`SEEDS_EXT_COUNTERPARTS`, already in `entries`)."""
    counterpart = SEEDS_EXT_COUNTERPARTS.get(group.id)
    if counterpart is None:
        return None
    entry = entries[counterpart]
    return {"seconds_per_run": entry["seconds_per_run"], "basis": "estimate",
            "rule": f"equal to {counterpart} (the same computation on seeds 1-3, {entry['basis']})"}


# ADR 0021: group of the kmh_confirm part (seeds 4-6) -> (its seeds-1-3 counterpart, mean wall time per run of the
# counterpart measured on the clean campaign, the log of the clean campaign holding its three `DONE elapsed` lines).
_CLEAN_MAIN_B = "results/campaign_fixed/campaign_part_2_gpu.log"
_CLEAN_NORM_A = "results/campaign_kmh_fixed/campaign_part_1_gpu__kmh_norm.log"
KMH_CONFIRM_COUNTERPARTS: dict[str, tuple[str, int, str]] = {
    "kmh_confirm/nq100/kmh_l2": ("kmh_norm/nq100/l2", 592, _CLEAN_NORM_A),  # 598, 590, 587 s
    "kmh_confirm/nq50/kmh_l2": ("kmh_norm/nq50/l2", 559, _CLEAN_NORM_A),  # 561, 557, 559 s
    "kmh_confirm/nq10/kmh_l2": ("kmh_norm/nq10/l2", 537, _CLEAN_NORM_A),  # 540, 535, 535 s
    "kmh_confirm/nq50/kmh_reference": ("p1/KMH/nq50", 574, _CLEAN_MAIN_B),  # 578, 570, 574 s
    "kmh_confirm/nq10/kmh_reference": ("p1/KMH/nq10", 543, _CLEAN_MAIN_B),  # 544, 548, 538 s
    "kmh_confirm/nq50/RandomSampling": ("p1/RandomSampling/nq50", 82, _CLEAN_MAIN_B),  # 81, 82, 83 s
    "kmh_confirm/nq10/RandomSampling": ("p1/RandomSampling/nq10", 56, _CLEAN_MAIN_B),  # 57, 56, 56 s
    "kmh_confirm/nq50/WAAL": ("p1/WAAL/nq50", 3589, _CLEAN_MAIN_B),  # 3571, 3597, 3598 s
    "kmh_confirm/nq10/WAAL": ("p1/WAAL/nq10", 3650, _CLEAN_MAIN_B),  # 3647, 3651, 3651 s
}


def confirm_estimate(group: RunGroup) -> dict[str, Any] | None:
    """Entry of a kmh_confirm group: its counterpart's clean-campaign wall time (`KMH_CONFIRM_COUNTERPARTS`)."""
    found = KMH_CONFIRM_COUNTERPARTS.get(group.id)
    if found is None:
        return None
    counterpart, seconds, log = found
    return {"seconds_per_run": seconds, "basis": "estimate",
            "rule": f"equal to {counterpart} (the same computation on seeds 1-3), mean wall time of its three runs "
                    f"measured on the clean campaign ({log})"}


# Budget folder -> measured paper-1 KMH run of that query size (same reference hierarchy, ImageNet features).
KMH_BY_BUDGET: dict[str, str] = {"nq100": "shared/kmh_nq100", "nq50": "p1/KMH/nq50", "nq10": "p1/KMH/nq10"}


def scaled_estimate(group: RunGroup, measured: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Entry of a never-run group estimated by budget scaling (`SCALED_PREFIXES`), or None."""
    for prefix, (reference, rule) in SCALED_PREFIXES.items():
        if group.id.startswith(prefix):
            kmh = KMH_BY_BUDGET[group.id[len(prefix):]]
            ratio = measured[kmh]["seconds_per_run"] / measured[KMH_BY_BUDGET["nq100"]]["seconds_per_run"]
            return {"seconds_per_run": round(measured[reference]["seconds_per_run"] * ratio), "basis": "estimate",
                    "rule": rule.format(kmh=kmh)}
    return None


def reference_estimate(group: RunGroup, measured: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Entry of a never-run group estimated from its measured reference (`REFERENCE_PREFIXES`), or None."""
    for prefix, (reference, rule) in REFERENCE_PREFIXES.items():
        if group.id.startswith(prefix):
            nq = group.id[len(prefix):].split("/", 1)[0]
            return {"seconds_per_run": measured[reference.format(nq=nq)]["seconds_per_run"], "basis": "estimate",
                    "rule": rule.format(nq=nq)}
    return None


def _clusters(group: RunGroup) -> tuple[int, ...]:
    params = json.loads((REPO_ROOT / group.params_json).read_text())
    return tuple(params[next(k for k in params if not k.startswith("_"))]["selection"]["hierarchy"]["n_clusters"])


def fit_estimate(group: RunGroup, groups: list[RunGroup], measured: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Entry of a never-run group estimated by the linear fit of `FIT_PREFIXES`, or None."""
    rule = next((r for prefix, r in FIT_PREFIXES.items() if group.id.startswith(prefix)), None)
    if rule is None:
        return None
    clusters = _clusters(group)
    points = [(sum(_clusters(g)), measured[g.id]["seconds_per_run"]) for g in groups
              if g.id.startswith(SWEEP_ID_PREFIX) and "__" not in g.id and g.id in measured
              and g.n_query == group.n_query and len(_clusters(g)) == len(clusters)]
    if len({x for x, _ in points}) < 2:
        raise SystemExit(f"{group.id}: fewer than two measured sweep shapes to fit (budget, levels)")
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points)
    total = sum(clusters)
    seconds = mean_y + slope * (total - mean_x)
    return {"seconds_per_run": round(seconds), "basis": "estimate", "rule": rule.format(total=total)}


def _old_dir(group: RunGroup) -> Path:
    for new, old in MEASURED_ROOTS.items():
        if group.dir_results.startswith(new + "/"):
            return REPO_ROOT / (old + group.dir_results[len(new):])
    raise ValueError(f"{group.id}: dir_results {group.dir_results!r} is under no known results root")


def _done_elapsed() -> dict[tuple[str, int], float]:
    done: dict[tuple[str, int], float] = {}
    for old in MEASURED_ROOTS.values():
        log = REPO_ROOT / old / "campaign.log"
        if log.is_file():
            for match in DONE_RE.finditer(log.read_text(errors="replace")):
                done[(match.group(1), int(match.group(2)))] = float(match.group(3))
    return done


def _total_time(group: RunGroup, seed: int) -> float | None:
    # NIL_*: the upper bound's n_init_labeled is the pool size, which differs between the two datasets.
    pattern = f"*/SEED_{seed}/NQ_{group.n_query}_NIL_*_NR_{group.n_round}_NE_*/{group.strategy_name}/log-dalmax.log"
    for log in sorted(_old_dir(group).glob(pattern)):
        match = TOTAL_TIME_RE.search(log.read_text(errors="replace"))
        if match:
            return float(match.group(1))
    return None


def measure(groups: list[RunGroup]) -> tuple[dict[str, dict[str, Any]], float]:
    """Measured entries (group id -> entry) and the median subprocess overhead used for the fallback."""
    done = _done_elapsed()
    raw: dict[str, list[tuple[float | None, float | None]]] = {}
    for group in groups:
        if group.strategy_name in ESTIMATE_RULES or any(
                group.id.startswith(p) for p in (*TWIN_PREFIXES, *FIT_PREFIXES, *REFERENCE_PREFIXES,
                                                 *SCALED_PREFIXES)) or group.id in SEEDS_EXT_COUNTERPARTS \
                or group.id in KMH_CONFIRM_COUNTERPARTS:
            continue
        raw[group.id] = [(done.get((group.id, s)), _total_time(group, s)) for s in group.seeds]
    overheads = [d - t for runs in raw.values() for d, t in runs if d is not None and t is not None]
    overhead = statistics.median(overheads) if overheads else 0.0
    measured: dict[str, dict[str, Any]] = {}
    for gid, runs in raw.items():
        per_run = [d if d is not None else t + overhead for d, t in runs if d is not None or t is not None]
        if len(per_run) != len(runs):
            continue  # not every seed was found: leave it to the caller's error
        measured[gid] = {
            "seconds_per_run": round(statistics.fmean(per_run)),
            "basis": "measured",
            "runs": len(per_run),
            "from_campaign_log": sum(d is not None for d, _ in runs),
        }
    return measured, overhead


def estimate(group: RunGroup, measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    family, factor, fixed, rule = ESTIMATE_RULES[group.strategy_name]
    if fixed is not None:
        seconds = fixed
    else:
        budget = group.id.rsplit("/", 1)[1]  # "nq100"
        refs = []
        for strategy in family:
            ref = "shared/random_nq100" if (strategy, budget) == ("RandomSampling", "nq100") else f"p1/{strategy}/{budget}"
            refs.append(measured[ref]["seconds_per_run"])
        seconds = factor * statistics.fmean(refs)
    return {"seconds_per_run": round(seconds), "basis": "estimate", "rule": rule}


def build() -> dict[str, Any]:
    groups = [g for path in MANIFESTS for g in load_manifest(REPO_ROOT / path, validate=False).groups]
    measured, overhead = measure(groups)
    entries: dict[str, dict[str, Any]] = {}
    for group in groups:
        twin = twin_estimate(group, measured)
        fit = fit_estimate(group, groups, measured)
        reference = reference_estimate(group, measured)
        scaled = scaled_estimate(group, measured)
        counterpart = counterpart_estimate(group, entries)
        confirm = confirm_estimate(group)
        if twin is not None:
            entries[group.id] = twin
        elif fit is not None:
            entries[group.id] = fit
        elif reference is not None:
            entries[group.id] = reference
        elif scaled is not None:
            entries[group.id] = scaled
        elif counterpart is not None:
            entries[group.id] = counterpart
        elif confirm is not None:
            entries[group.id] = confirm
        elif group.strategy_name in ESTIMATE_RULES:
            entries[group.id] = estimate(group, measured)
        elif group.id in measured:
            entries[group.id] = measured[group.id]
        else:
            raise SystemExit(f"{group.id}: no measured run of the first execution found under {_old_dir(group)}")
    return {
        "_comment": "Estimated GPU seconds per run of every group of files_config/campaign/manifest.json and "
                    "manifest_kmh_sweep.json (ADR 0016). Generated by scripts/campaign/build_gpu_time_estimates.py "
                    "(see its docstring for the method and caveats): 'measured' = mean wall time per run of the same "
                    "group in the first execution (dataset with duplicates, larger pool and test set, earlier code "
                    "for the ADR 0014 strategies); 'estimate' = rule for a strategy or group never run on the GPU. For "
                    "balancing the two GPU parts and budgeting, not a performance claim.",
        "unit": "seconds per run (trainer subprocess wall time)",
        "subprocess_overhead_seconds": round(overhead),
        "groups": entries,
    }


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed table differs")
    args = parser.parse_args()
    if not all((REPO_ROOT / old / "campaign.log").is_file() for old in MEASURED_ROOTS.values()):
        print("the first-execution trees (results/campaign/, results/campaign_kmh/) are not here: nothing to do")
        return 0
    text = render()
    path = REPO_ROOT / OUT
    if args.check:
        if not path.is_file() or path.read_text() != text:
            print(f"DRIFT: {OUT}")
            return 1
        return 0
    path.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
