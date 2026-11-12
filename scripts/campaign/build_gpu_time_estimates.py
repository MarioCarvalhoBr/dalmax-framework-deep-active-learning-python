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
- **estimate** groups: the eight DeepAL+ strategies of ADR 0015, never executed on the GPU. Rules
  (author's brief of 2026-11-11, not measurements): `LossPredictionLoss`, `BadgeSampling` and
  `KCenterGreedyPCA` cost about a plain uncertainty strategy (the mean of `LeastConfidence`,
  `MarginSampling` and `EntropySampling` at the same budget); `VarRatio` and `MeanSTD` about an
  MC-dropout strategy (mean of the four at the same budget); `CEALSampling` 1.25 times a plain one;
  `WAAL` about 45 minutes and `VAAL` (VAE images 32 x 32) about 50 minutes per run (FLOP estimates).

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
# strategy -> (reference family, multiplier, fixed seconds or None, rule text)
ESTIMATE_RULES: dict[str, tuple[tuple[str, ...], float, float | None, str]] = {
    "LossPredictionLoss": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "BadgeSampling": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "KCenterGreedyPCA": (PLAIN, 1.0, None, "about a plain uncertainty strategy at the same budget"),
    "CEALSampling": (PLAIN, 1.25, None, "1.25 x a plain uncertainty strategy at the same budget"),
    "VarRatio": (MC_DROPOUT, 1.0, None, "about an MC-dropout strategy at the same budget"),
    "MeanSTD": (MC_DROPOUT, 1.0, None, "about an MC-dropout strategy at the same budget"),
    "WAAL": ((), 1.0, 45 * 60.0, "about 45 minutes per run (FLOP estimate)"),
    "VAAL": ((), 1.0, 50 * 60.0, "about 50 minutes per run at VAE image size 32 (FLOP estimate)"),
}


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
        if group.strategy_name in ESTIMATE_RULES:
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
        if group.strategy_name in ESTIMATE_RULES:
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
                    "for the ADR 0014 strategies); 'estimate' = rule for a strategy never run on the GPU. For "
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
