"""Campaign manifests: load, validate, expand to jobs, run, verify.

`files_config/campaign/manifest.json` is the single source of truth for the
clean re-execution of everything the three papers need on
`DATA/daninhas_full_fixed` (see `.specs/experiments/campaign.md`, ADR 0008 and
ADR 0016); `manifest_kmh_sweep.json` is the paper-1 KMH sweep (ADR 0012). Each
*run group* is one distinct computation (params JSON + strategy + n_query +
n_init_labeled + n_round) swept over the manifest's seeds; a table row of any
paper that needs the same computation points to that one group (directly, or
through an *alias* entry that documents why two differently-spelled configs are
the same computation). Nothing runs twice.

Two orthogonal partitions select groups (`--part`):

- the *semantic parts* `PARTS` (`paper1`, `upper_bound`, `rnhal`, `texhal`,
  `texhal_q17`, `hier_transfer`, `control`, `kmh_nq100`, `kmh_nq50`, `kmh_nq10`;
  comma-separated, or `all`), executed in the fixed `PHASE_LABELS` order;
- the *GPU parts* `GPU_PARTS` (`part_1_gpu`, `part_2_gpu`, ADR 0016): every group of
  the two full-scale manifests carries a `gpu_part` and a `gpu_stage`; selecting a
  GPU part orders its jobs by stage (cheap table-critical groups first, the very
  long baselines last). The two kinds cannot be mixed in one `--part`.

CLI (`python -m dalmax.campaign <cmd>`; `--manifest` accepts a comma-separated
list, e.g. the main manifest and the KMH sweep, whose jobs are merged into one
plan: by GPU stage for a GPU part, manifest by manifest otherwise):

- `list`: print the job table, the counts and the estimated GPU hours per part and
  per GPU part (`gpu_time_estimates` of the manifest, when it has one).
- `run --part ... [--micro] [--dry-run] [--skip-existing|--no-skip-existing]
  [--device cuda]`: execute the jobs sequentially through `tools/trainer.py`
  (subprocess), logging to `<results_root>/campaign.log` and
  `<results_root>/failures.log` of the job's manifest, continuing after a failure.
- `verify [--part ...] [--micro]`: per job OK / INCOMPLETE / MISSING, reusing
  `dalmax.reporting.leaf_check.check_leaf` (plus the manifest's
  `required_leaf_files`), and the seed-consistency audit over every selected job.
  Exit code: 0 all OK; 1 an incomplete/missing job or a FAIL audit; 2 only audit
  WARNs (a run whose initial-labeled-set log line is missing/unreadable).
- Every subcommand accepts `--exclude-strategy A,B` (e.g. the two adversarial
  baselines, far too slow for a CPU smoke run): jobs of those strategies are
  dropped from the list/run/verify/audit. The manifest still contains them.

Semantic execution order (fixed, independent of `--part`): paper 1 at
n_query=100 first (the shared canonical runs -- KMH and RandomSampling, which
papers 2/3 reuse -- land early), then rnhal, texhal, texhal_q17, hier_transfer,
then paper 1 at n_query=50 and 10, then the upper bound.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import FULL_SUPERVISED_STRATEGY, STRATEGY_ARGS_KEYS
from dalmax.query_strategies.registry import (
    GENERIC_REPRESENTATION_NAME,
    REPRESENTATION_PRESET_Q,
    REPRESENTATION_PRESETS,
)
from dalmax.reporting.leaf_check import ROUND_HISTORY_FILES, check_leaf
from dalmax.seeding import CUBLAS_WORKSPACE_CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = "files_config/campaign/manifest.json"
DEFAULT_MICRO_MANIFEST = "files_config/campaign/manifest_micro.json"
TRAINER_SCRIPT = "tools/trainer.py"

# Layout/ownership rule: a run group lives under the FIRST paper that consumes
# it (paper 1 < 2 < 3) and its `used_by` lists every consumer; e.g. paper 1's
# `KMH@100` is also the "w/o representation module" row of papers 2 and 3.
PARTS: tuple[str, ...] = ("paper1", "upper_bound", "rnhal", "texhal", "texhal_q17", "hier_transfer",
                          "control", "kmh_nq100", "kmh_nq50", "kmh_nq10")

# The two GPU parts (ADR 0016): one per Colab account, balanced by estimated GPU time.
GPU_PARTS: tuple[str, ...] = ("part_1_gpu", "part_2_gpu")

# Execution phases (see module docstring). Index = order of execution.
PHASE_LABELS: tuple[str, ...] = (
    "paper1_nq100",
    "rnhal",
    "texhal",
    "texhal_q17",
    "hier_transfer",
    "paper1_other_nq",
    "upper_bound",
    "control",
    "kmh_nq100",
    "kmh_nq50",
    "kmh_nq10",
)


class CampaignError(ValueError):
    """Raised for any invalid manifest / campaign request (fail fast)."""


# --- data model ---------------------------------------------------------------


@dataclass(frozen=True)
class RunGroup:
    """One distinct computation, swept over `seeds`."""

    id: str
    part: str
    params_json: str
    strategy_name: str
    n_query: int
    n_init_labeled: int
    n_round: int
    seeds: tuple[int, ...]
    dir_results: str
    used_by: tuple[dict[str, str], ...]
    note: str = ""
    gpu_part: str | None = None  # one of GPU_PARTS, or None (manifest without GPU parts)
    gpu_stage: int | None = None  # execution stage inside the GPU part (lower runs first)


@dataclass(frozen=True)
class Alias:
    """A differently-spelled config that is the SAME computation as `canonical`."""

    id: str
    canonical: str
    params_json: str
    strategy_name: str
    reason: str


@dataclass(frozen=True)
class Manifest:
    name: str
    dataset_name: str
    results_root: str
    seeds: tuple[int, ...]
    primary_nq: int
    groups: tuple[RunGroup, ...]
    aliases: tuple[Alias, ...]
    tables: dict[str, Any]
    path: str = ""
    # Files every leaf of this manifest must carry on top of `leaf_check.REQUIRED_FILES`
    # (ADR 0016: the per-round history files for the campaign on daninhas_full_fixed).
    required_leaf_files: tuple[str, ...] = ()
    # Repo-relative path of the GPU-time estimate table (`load_estimates`), or "" for none.
    gpu_time_estimates: str = ""

    def group_by_id(self) -> dict[str, RunGroup]:
        return {g.id: g for g in self.groups}

    def alias_by_id(self) -> dict[str, Alias]:
        return {a.id: a for a in self.aliases}


@dataclass(frozen=True)
class Job:
    """One `(group, seed)` execution."""

    group: RunGroup
    seed: int

    @property
    def run_id(self) -> str:
        return self.group.id

    def label(self) -> str:
        return f"{self.group.id} seed={self.seed}"


# --- loading ------------------------------------------------------------------


def _to_group(raw: dict[str, Any], default_seeds: tuple[int, ...]) -> RunGroup:
    try:
        return RunGroup(
            id=raw["id"],
            part=raw["part"],
            params_json=raw["params_json"],
            strategy_name=raw["strategy_name"],
            n_query=int(raw["n_query"]),
            n_init_labeled=int(raw["n_init_labeled"]),
            n_round=int(raw["n_round"]),
            seeds=tuple(int(s) for s in raw.get("seeds", default_seeds)),
            dir_results=raw["dir_results"],
            used_by=tuple(dict(u) for u in raw.get("used_by", [])),
            note=raw.get("note", ""),
            gpu_part=raw.get("gpu_part"),
            gpu_stage=int(raw["gpu_stage"]) if raw.get("gpu_stage") is not None else None,
        )
    except KeyError as exc:
        raise CampaignError(f"run group {raw.get('id')!r} is missing key {exc}") from exc


def load_manifest(path: str | Path, *, validate: bool = True) -> Manifest:
    """Parse (and by default validate) a campaign manifest JSON."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read manifest {p}: {exc}") from exc

    seeds = tuple(int(s) for s in raw.get("seeds", (1, 2, 3)))
    manifest = Manifest(
        name=raw.get("name", p.stem),
        dataset_name=raw.get("dataset_name", "DANINHAS"),
        results_root=raw["results_root"],
        seeds=seeds,
        primary_nq=int(raw.get("primary_nq", 100)),
        groups=tuple(_to_group(g, seeds) for g in raw["groups"]),
        aliases=tuple(
            Alias(
                id=a["id"],
                canonical=a["canonical"],
                params_json=a["params_json"],
                strategy_name=a["strategy_name"],
                reason=a.get("reason", ""),
            )
            for a in raw.get("aliases", [])
        ),
        tables=raw.get("tables", {}),
        path=str(p),
        required_leaf_files=tuple(raw.get("required_leaf_files", ())),
        gpu_time_estimates=raw.get("gpu_time_estimates", ""),
    )
    if validate:
        validate_manifest(manifest)
    return manifest


# --- effective-config signature / redundancy check --------------------------


def _resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def effective_signature(
    params_json: str,
    dataset_name: str,
    strategy_name: str,
    n_query: int,
    n_init_labeled: int,
    n_round: int,
) -> str:
    """Canonical string identifying *the computation* a run performs.

    Two runs with the same signature are the same computation (given the
    same seed): the signature is built from the fully-resolved
    `ExperimentConfig` minus everything that is only bookkeeping
    (`dir_results`, the params file path, the seed, the device), with the
    strategy reduced to its behavior:

    - legacy strategies (`RandomSampling`, ...): the strategy name; the
      params JSON's embedding/selection blocks are irrelevant to them and
      excluded;
    - representation strategies: the preset names (`SSRAEKmeansHCSampling`,
      `VCTexKmeansHCSampling`, ...) are resolved to `(extractor, pinned q,
      selection method)` exactly like `build_strategy` does (the params
      JSON's `embedding.q` is ignored by presets), and the generic
      `RepresentationStrategy` reads `embedding`/`selection` verbatim -- so a
      preset and its generic equivalent get the same signature iff they
      build the same provider + selection.

    Per-strategy hyperparameters (`strategy_args`, ADR 0015) are reduced to
    the one block the strategy reads (`STRATEGY_ARGS_KEYS`; none for every
    other strategy), so two params files that differ only in another
    strategy's block are the same computation. For the groups executed
    before ADR 0015 this changes no dedup decision: their params have no
    `strategy_args`, so every block was the default anyway.
    """
    config = load_experiment_config(
        _resolve_path(params_json),
        dataset_name,
        strategy_name=strategy_name,
        seed=0,
        n_init_labeled=n_init_labeled,
        n_query=n_query,
        n_round=n_round,
        dir_results="_signature/",
        device="cpu",
    )
    cfg = to_dict(config)
    dataset = dict(cfg["dataset"])
    embedding = dataset.pop("embedding")
    selection = dataset.pop("selection")
    strategy_args = dataset.pop("strategy_args")
    own_args_key = STRATEGY_ARGS_KEYS.get(strategy_name)
    if own_args_key is not None:
        dataset["strategy_args"] = {own_args_key: strategy_args[own_args_key]}

    behavior: dict[str, Any]
    if strategy_name in REPRESENTATION_PRESETS:
        extractor, method = REPRESENTATION_PRESETS[strategy_name]
        behavior = {
            "kind": "representation",
            "extractor": extractor,
            "q": REPRESENTATION_PRESET_Q[extractor],
            "variant": embedding["variant"],
            "selection_method": method,
            "hierarchy": selection.get("hierarchy") if method == "hierarchical" else None,
        }
    elif strategy_name == GENERIC_REPRESENTATION_NAME:
        method = selection["method"]
        behavior = {
            "kind": "representation",
            "extractor": embedding["extractor"],
            "q": embedding["q"],
            "variant": embedding["variant"],
            "selection_method": method,
            "hierarchy": selection.get("hierarchy") if method == "hierarchical" else None,
        }
    else:
        behavior = {"kind": "legacy", "strategy": strategy_name}

    payload = {
        "dataset": dataset,
        "behavior": behavior,
        "n_query": n_query,
        "n_init_labeled": n_init_labeled,
        "n_round": n_round,
    }
    return json.dumps(_normalize(payload), sort_keys=True)


def _normalize(obj: Any) -> Any:
    """Tuples -> lists (JSON-native), recursively, so signatures compare equal."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def group_signature(manifest: Manifest, group: RunGroup) -> str:
    return effective_signature(
        group.params_json,
        manifest.dataset_name,
        group.strategy_name,
        group.n_query,
        group.n_init_labeled,
        group.n_round,
    )


def find_redundant_groups(manifest: Manifest) -> list[tuple[str, str]]:
    """Pairs of group ids that are the same computation -- no exceptions
    (the former "duplicate by decision" exemption was reversed, ADR 0009)."""
    seen: dict[str, RunGroup] = {}
    redundant: list[tuple[str, str]] = []
    for group in manifest.groups:
        sig = group_signature(manifest, group)
        other = seen.get(sig)
        if other is None:
            seen[sig] = group
            continue
        redundant.append((other.id, group.id))
    return redundant


# --- validation ---------------------------------------------------------------


def derive_used_by(tables: dict[str, Any], manifest_aliases: dict[str, Alias]) -> dict[str, list[dict[str, str]]]:
    """`{canonical group id: [{paper, table, row}, ...]}` derived from `tables`.

    Rows that use `best_of` (computed at report time) are attributed to no
    single group. Alias ids resolve to their canonical group.
    """
    used: dict[str, list[dict[str, str]]] = {}
    for paper, paper_tables in tables.items():
        for table in paper_tables.get("tables", []):
            for row in table["rows"]:
                run = row.get("run")
                if run is None:
                    continue
                canonical = manifest_aliases[run].canonical if run in manifest_aliases else run
                used.setdefault(canonical, []).append(
                    {"paper": paper, "table": table["id"], "row": row["label"]}
                )
    return used


def _canon_entry(entry: dict[str, str]) -> tuple[str, str, str]:
    return (entry["paper"], entry["table"], entry["row"])


def validate_manifest(manifest: Manifest) -> None:
    """Fail fast on any structural or redundancy problem."""
    groups = manifest.group_by_id()
    if len(groups) != len(manifest.groups):
        raise CampaignError("duplicate run-group ids in manifest")
    aliases = manifest.alias_by_id()
    if len(aliases) != len(manifest.aliases):
        raise CampaignError("duplicate alias ids in manifest")
    clash = set(groups) & set(aliases)
    if clash:
        raise CampaignError(f"ids used both as group and alias: {sorted(clash)}")

    unknown_files = sorted(set(manifest.required_leaf_files) - set(ROUND_HISTORY_FILES))
    if unknown_files:
        raise CampaignError(f"required_leaf_files: unknown {unknown_files}; known: {ROUND_HISTORY_FILES}")
    leaves: dict[tuple, str] = {}
    for group in manifest.groups:
        if group.part not in PARTS:
            raise CampaignError(f"{group.id}: part {group.part!r} not in {PARTS}")
        if group.gpu_part is not None and group.gpu_part not in GPU_PARTS:
            raise CampaignError(f"{group.id}: gpu_part {group.gpu_part!r} not in {GPU_PARTS}")
        if (group.gpu_part is None) != (group.gpu_stage is None):
            raise CampaignError(f"{group.id}: gpu_part and gpu_stage must be set together")
        if not group.seeds:
            raise CampaignError(f"{group.id}: no seeds")
        # Every params file must load through the real config loader.
        try:
            group_signature(manifest, group)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise CampaignError(f"{group.id}: params_json {group.params_json!r} failed to load: {exc}") from exc
        key = (
            group.dir_results.rstrip("/"),
            group.strategy_name,
            group.n_query,
            group.n_init_labeled,
            group.n_round,
        )
        if key in leaves:
            raise CampaignError(f"{group.id} and {leaves[key]} would write the same results leaf")
        leaves[key] = group.id

    redundant = find_redundant_groups(manifest)
    if redundant:
        raise CampaignError(f"redundant run groups (same computation): {redundant}")

    for alias in manifest.aliases:
        canonical = groups.get(alias.canonical)
        if canonical is None:
            raise CampaignError(f"alias {alias.id}: canonical {alias.canonical!r} is not a run group")
        try:
            alias_sig = effective_signature(
                alias.params_json,
                manifest.dataset_name,
                alias.strategy_name,
                canonical.n_query,
                canonical.n_init_labeled,
                canonical.n_round,
            )
        except Exception as exc:  # noqa: BLE001
            raise CampaignError(f"alias {alias.id}: params_json failed to load: {exc}") from exc
        if alias_sig != group_signature(manifest, canonical):
            raise CampaignError(
                f"alias {alias.id} is NOT the same computation as {alias.canonical} "
                "(effective configs differ) -- do not dedupe it"
            )

    # Table rows must resolve; used_by must match the tables.
    for paper_tables in manifest.tables.values():
        for table in paper_tables.get("tables", []):
            for row in table["rows"]:
                if "run" in row:
                    if row["run"] not in groups and row["run"] not in aliases:
                        raise CampaignError(f"table {table['id']} row {row['label']!r}: unknown run {row['run']!r}")
                elif "best_of" in row:
                    for cand in row["best_of"]["candidates"]:
                        if cand["run"] not in groups and cand["run"] not in aliases:
                            raise CampaignError(f"table {table['id']}: best_of candidate {cand['run']!r} unknown")
                else:
                    raise CampaignError(f"table {table['id']} row {row['label']!r}: needs 'run' or 'best_of'")

    derived = derive_used_by(manifest.tables, aliases)
    for group in manifest.groups:
        want = sorted(_canon_entry(e) for e in derived.get(group.id, []))
        have = sorted(_canon_entry(e) for e in group.used_by)
        if want != have:
            raise CampaignError(f"{group.id}: used_by does not match the tables section (regenerate the manifest)")


# --- expansion ----------------------------------------------------------------


def phase_index(group: RunGroup, primary_nq: int) -> int:
    """Execution phase of a group (index into `PHASE_LABELS`); `primary_nq` is
    the manifest's paper-1 primary budget (the one papers 2/3 compare against)."""
    if group.part == "paper1":
        return PHASE_LABELS.index("paper1_nq100" if group.n_query == primary_nq else "paper1_other_nq")
    return PHASE_LABELS.index(group.part)


@dataclass(frozen=True)
class Selection:
    """Which groups a command acts on: semantic `parts`, or (if set) `gpu_parts`."""

    parts: tuple[str, ...] = PARTS
    gpu_parts: tuple[str, ...] | None = None

    def selects(self, group: RunGroup) -> bool:
        if self.gpu_parts is not None:
            return group.gpu_part in self.gpu_parts
        return group.part in self.parts


def parse_parts(spec: str) -> tuple[str, ...]:
    """`"all"` or a comma-separated subset of `PARTS` (semantic parts only)."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise CampaignError("empty --part")
    if "all" in names:
        return PARTS
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise CampaignError(f"unknown part(s) {bad}; valid: {', '.join(PARTS)}, {', '.join(GPU_PARTS)} or 'all'")
    return tuple(n for n in PARTS if n in names)


def parse_selection(spec: str) -> Selection:
    """`--part`: `"all"`, semantic parts, or GPU parts (`part_1_gpu`, `part_2_gpu`); never both kinds."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    gpu = [n for n in names if n in GPU_PARTS]
    if not gpu:
        return Selection(parse_parts(spec))
    if len(gpu) != len(names):
        raise CampaignError(f"--part {spec!r}: GPU parts ({', '.join(GPU_PARTS)}) cannot be mixed with other parts")
    return Selection(PARTS, tuple(n for n in GPU_PARTS if n in gpu))


def parse_exclude_strategies(spec: str | None, manifests: Manifest | Sequence[Manifest]) -> frozenset[str]:
    """Comma-separated strategy names to drop; each must exist in one of the manifests (fail fast)."""
    names = frozenset(s.strip() for s in (spec or "").split(",") if s.strip())
    pool = [manifests] if isinstance(manifests, Manifest) else list(manifests)
    known = {g.strategy_name for m in pool for g in m.groups}
    bad = sorted(names - known)
    if bad:
        raise CampaignError(f"--exclude-strategy: unknown strategy {bad}; manifest has: {sorted(known)}")
    return names


def expand_jobs(
    manifest: Manifest,
    parts: Sequence[str] | Selection = PARTS,
    seeds: Sequence[int] | None = None,
    exclude_strategies: Collection[str] = (),
) -> list[Job]:
    """All `(group, seed)` jobs of the selection, in execution order. `parts` is a
    sequence of semantic parts or a `Selection` (a GPU-part selection orders by
    `gpu_stage` first). `seeds` restricts to a subset of each group's seeds (smoke
    runs; default: all); groups whose `strategy_name` is in `exclude_strategies`
    are dropped."""
    selection = parts if isinstance(parts, Selection) else Selection(tuple(parts))
    by_stage = selection.gpu_parts is not None
    indexed = [
        ((g.gpu_stage or 0) if by_stage else 0, phase_index(g, manifest.primary_nq), i, g)
        for i, g in enumerate(manifest.groups)
        if selection.selects(g) and g.strategy_name not in exclude_strategies
    ]
    indexed.sort(key=lambda t: t[:3])
    return [
        Job(g, seed) for *_, g in indexed for seed in g.seeds if seeds is None or seed in seeds
    ]


@dataclass(frozen=True)
class Segment:
    """Consecutive jobs of one manifest inside an execution plan."""

    manifest: Manifest
    jobs: tuple[Job, ...]


def plan_jobs(
    manifests: Sequence[Manifest],
    selection: Selection,
    seeds: Sequence[int] | None = None,
    exclude_strategies: Collection[str] = (),
) -> list[Segment]:
    """The jobs of several manifests as one ordered plan of per-manifest segments.

    For a GPU-part selection the jobs are merged by `gpu_stage` (ties: manifest
    order), so e.g. the KMH sweep (stage 3) runs between the cheap groups and the
    long baselines of the main manifest; otherwise manifest after manifest.
    """
    keyed: list[tuple[tuple[int, int, int], int, Job]] = []
    for mi, manifest in enumerate(manifests):
        for pos, job in enumerate(expand_jobs(manifest, selection, seeds, exclude_strategies)):
            stage = (job.group.gpu_stage or 0) if selection.gpu_parts is not None else 0
            keyed.append(((stage, mi, pos), mi, job))
    keyed.sort(key=lambda t: t[0])
    segments: list[tuple[int, list[Job]]] = []
    for _, mi, job in keyed:
        if segments and segments[-1][0] == mi:
            segments[-1][1].append(job)
        else:
            segments.append((mi, [job]))
    return [Segment(manifests[mi], tuple(jobs)) for mi, jobs in segments]


def count_by_part(jobs: Sequence[Job]) -> dict[str, int]:
    counts = dict.fromkeys(PARTS, 0)
    for job in jobs:
        counts[job.group.part] += 1
    return counts


def count_by_gpu_part(jobs: Sequence[Job]) -> dict[str, int]:
    """Jobs per GPU part (`None` = groups without one)."""
    counts: dict[str, int] = dict.fromkeys(GPU_PARTS, 0)
    for job in jobs:
        key = job.group.gpu_part or "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- GPU-time estimates -------------------------------------------------------


def load_estimates(manifest: Manifest) -> dict[str, float]:
    """`{group id: estimated seconds per run}` from the manifest's `gpu_time_estimates`
    table (`{}` when the manifest has none). The table format is documented in
    `.specs/experiments/campaign.md` ("GPU-time estimates")."""
    if not manifest.gpu_time_estimates:
        return {}
    path = _resolve_path(manifest.gpu_time_estimates)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read GPU-time estimates {path}: {exc}") from exc
    return {gid: float(entry["seconds_per_run"]) for gid, entry in raw["groups"].items()}


def estimated_hours(jobs: Sequence[Job], estimates: dict[str, float]) -> tuple[float, int]:
    """`(hours, jobs without an estimate)` for `jobs`."""
    seconds = sum(estimates.get(j.group.id, 0.0) for j in jobs)
    unknown = sum(1 for j in jobs if j.group.id not in estimates)
    return seconds / 3600, unknown


# --- paths / skip-existing ----------------------------------------------------


def leaf_glob(group: RunGroup, seed: int) -> str:
    """Glob (relative to the group's `dir_results`) matching this job's leaf dir (any dataset folder / n_epoch)."""
    return (f"*/SEED_{seed}/NQ_{group.n_query}_NIL_{group.n_init_labeled}_NR_{group.n_round}_NE_*/"
            f"{group.strategy_name}")


def find_leaves(group: RunGroup, seed: int, *, base: Path | None = None) -> list[Path]:
    """Every existing leaf directory of a job (any dataset folder / n_epoch), sorted.

    `base` overrides the directory `dir_results` is resolved against (default
    the repo root); used by the report/tests to point at another tree.
    """
    root = (base if base is not None else REPO_ROOT) / group.dir_results
    if not root.is_dir():
        return []
    return sorted(root.glob(leaf_glob(group, seed)))


def find_leaf(group: RunGroup, seed: int, *, base: Path | None = None) -> Path | None:
    """The first existing leaf directory of a job (see `find_leaves`), or None."""
    matches = find_leaves(group, seed, base=base)
    return matches[0] if matches else None


def job_done(job: Job, required_extra: tuple[str, ...] = ()) -> bool:
    """True iff the job's leaf carries the FULL artifact set (`check_leaf`, plus the
    manifest's `required_leaf_files` passed as `required_extra`), so a run killed
    mid-report is re-run on resume instead of being skipped. (`results.json` is
    written last by the reporter.)"""
    leaf = find_leaf(job.group, job.seed)
    if leaf is None:
        return False
    missing, _warnings = check_leaf(leaf, required_extra)
    return not missing


# --- running ------------------------------------------------------------------


def build_command(job: Job, manifest: Manifest, device: str, python: str | None = None) -> list[str]:
    g = job.group
    return [
        python or sys.executable,
        TRAINER_SCRIPT,
        "--params_json", g.params_json,
        "--dataset_name", manifest.dataset_name,
        "--strategy_name", g.strategy_name,
        "--n_query", str(g.n_query),
        "--n_init_labeled", str(g.n_init_labeled),
        "--n_round", str(g.n_round),
        "--seed", str(job.seed),
        "--dir_results", g.dir_results,
        "--device", device,
    ]


def build_env(device: str) -> dict[str, str]:
    """Environment of every trainer subprocess (headless matplotlib, cuBLAS determinism, device pin)."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    # Determinism: seed_everything also sets this default inside the run; setting
    # it in the subprocess env guarantees it precedes the first cuBLAS call.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def _default_executor(cmd: list[str], env: dict[str, str]) -> int:
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


@dataclass
class RunSummary:
    executed: int = 0
    skipped: int = 0
    failed: int = 0
    dry: int = 0


def run_jobs(
    jobs: Sequence[Job],
    manifest: Manifest,
    *,
    device: str,
    skip_existing: bool = True,
    dry_run: bool = False,
    executor: Callable[[list[str], dict[str, str]], int] = _default_executor,
    echo: Callable[[str], None] = print,
) -> RunSummary:
    """Execute jobs (all of `manifest`) sequentially; a failing job is logged and the batch continues."""
    root = REPO_ROOT / manifest.results_root
    campaign_log = root / "campaign.log"
    failures_log = root / "failures.log"
    summary = RunSummary()
    env = build_env(device)
    total = len(jobs)

    def log(msg: str) -> None:
        line = f"{_stamp()} {msg}"
        echo(line)
        if not dry_run:
            _append(campaign_log, line)

    log(f"CAMPAIGN START manifest={manifest.name} jobs={total} device={device} skip_existing={skip_existing}")
    for n, job in enumerate(jobs, start=1):
        head = f"[{n}/{total}] {job.label()}"
        if skip_existing and job_done(job, manifest.required_leaf_files):
            summary.skipped += 1
            log(f"{head} SKIP (complete leaf exists)")
            continue
        cmd = build_command(job, manifest, device)
        if dry_run:
            summary.dry += 1
            log(f"{head} DRY-RUN CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '-')} {' '.join(cmd)}")
            continue
        log(f"{head} START")
        t0 = time.time()
        code = executor(cmd, env)
        elapsed = time.time() - t0
        if code != 0:
            summary.failed += 1
            log(f"{head} FAILED exit={code} elapsed={elapsed:.0f}s")
            _append(failures_log, f"{_stamp()} FAILED {job.label()} exit={code}")
        else:
            summary.executed += 1
            log(f"{head} DONE elapsed={elapsed:.0f}s")
    log(
        f"CAMPAIGN END executed={summary.executed} skipped={summary.skipped} "
        f"failed={summary.failed} dry_run={summary.dry}"
    )
    return summary


# --- verify -------------------------------------------------------------------


@dataclass(frozen=True)
class JobStatus:
    job: Job
    status: str  # OK | INCOMPLETE | MISSING
    missing_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def verify_job(job: Job, *, base: Path | None = None, required_extra: tuple[str, ...] = ()) -> JobStatus:
    leaf = find_leaf(job.group, job.seed, base=base)
    if leaf is None:
        return JobStatus(job, "MISSING")
    missing, warnings = check_leaf(leaf, required_extra)
    if missing:
        return JobStatus(job, "INCOMPLETE", tuple(missing), tuple(warnings))
    return JobStatus(job, "OK", (), tuple(warnings))


# --- seed-consistency audit ---------------------------------------------------

INITIAL_IDXS_RE = re.compile(r"Initial labeled idxs \(sorted\): (\[[^\]]*\])")

# Runs that label the whole pool at initialization are excluded from the audit.
AUDIT_EXCLUDED_STRATEGIES: frozenset[str] = frozenset({FULL_SUPERVISED_STRATEGY})


@dataclass(frozen=True)
class SeedAudit:
    """Initial-labeled-set consistency of every run of one seed."""

    seed: int
    status: str  # PASS | WARN | FAIL | NO_DATA
    n_runs: int
    reference: str | None = None
    offenders: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()


def read_initial_idxs(leaf: Path) -> list[int] | None:
    """Parse `Initial labeled idxs (sorted): [...]` from `<leaf>/log-dalmax.log`."""
    log = leaf / "log-dalmax.log"
    if not log.is_file():
        return None
    match = INITIAL_IDXS_RE.search(log.read_text(errors="replace"))
    return json.loads(match.group(1)) if match else None


def audit_seed_consistency(jobs: Sequence[Job], *, base: Path | None = None) -> list[SeedAudit]:
    """For each seed, every run (all strategies/configs, `FullSupervised`
    excluded) must have started from the IDENTICAL initial labeled set -- the
    fair-comparison premise of the whole campaign."""
    by_seed: dict[int, list[tuple[str, list[int] | None]]] = {}
    for job in jobs:
        if job.group.strategy_name in AUDIT_EXCLUDED_STRATEGIES:
            continue
        leaf = find_leaf(job.group, job.seed, base=base)
        if leaf is None:
            continue  # not run yet: `verify` reports it as MISSING
        by_seed.setdefault(job.seed, []).append((job.group.id, read_initial_idxs(leaf)))

    seeds = sorted({j.seed for j in jobs if j.group.strategy_name not in AUDIT_EXCLUDED_STRATEGIES})
    audits: list[SeedAudit] = []
    for seed in seeds:
        runs = by_seed.get(seed, [])
        readable = [(rid, idxs) for rid, idxs in runs if idxs is not None]
        unreadable = tuple(rid for rid, idxs in runs if idxs is None)
        if not readable:
            # Runs exist but none can be compared (missing/unreadable log): WARN, not silence.
            status = "WARN" if unreadable else "NO_DATA"
            audits.append(SeedAudit(seed, status, len(runs), unreadable=unreadable))
            continue
        reference_id, reference = readable[0]
        offenders = tuple(rid for rid, idxs in readable if idxs != reference)
        status = "FAIL" if offenders else ("WARN" if unreadable else "PASS")
        audits.append(SeedAudit(seed, status, len(readable), reference_id, offenders, unreadable))
    return audits


def render_seed_audit(audits: Sequence[SeedAudit]) -> str:
    lines = ["Seed-consistency audit (initial labeled set identical across all strategies/configs):"]
    for a in audits:
        line = f"  seed {a.seed}: {a.status} ({a.n_runs} runs compared"
        line += f", reference {a.reference})" if a.reference else ")"
        lines.append(line)
        for rid in a.offenders:
            lines.append(f"    OFFENDER (different initial set): {rid}")
        for rid in a.unreadable:
            lines.append(f"    unreadable log-dalmax.log / no initial-idxs line: {rid}")
    return "\n".join(lines)


def render_verify(statuses: Sequence[JobStatus]) -> str:
    lines: list[str] = []
    for st in statuses:
        extra = ""
        if st.missing_files:
            extra = f" (missing {', '.join(st.missing_files)})"
        if st.warnings:
            extra += f" [warnings: {'; '.join(st.warnings)}]"
        lines.append(f"{st.status:<10} {st.job.label()}{extra}")
    ok = sum(s.status == "OK" for s in statuses)
    inc = sum(s.status == "INCOMPLETE" for s in statuses)
    mis = sum(s.status == "MISSING" for s in statuses)
    lines.append(f"Totals: OK={ok} INCOMPLETE={inc} MISSING={mis} (of {len(statuses)} jobs)")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------


def render_job_table(jobs: Sequence[Job], primary_nq: int, estimates: dict[str, float] | None = None) -> str:
    """The job table of one manifest, then runs / groups / estimated hours per part and per GPU part."""
    estimates = estimates or {}
    lines = [f"{'#':>4}  {'phase':<16} {'gpu part':<10} {'stg':>3} {'run id':<44} {'strategy':<24} "
             f"{'nq':>4} {'nil':>5} {'nr':>3} seed {'est h':>6}"]
    for n, job in enumerate(jobs, start=1):
        g = job.group
        est = f"{estimates[g.id] / 3600:6.2f}" if g.id in estimates else f"{'-':>6}"
        lines.append(
            f"{n:>4}  {PHASE_LABELS[phase_index(g, primary_nq)]:<16} {g.gpu_part or '-':<10} "
            f"{g.gpu_stage if g.gpu_stage is not None else '-':>3} {g.id:<44} {g.strategy_name:<24} "
            f"{g.n_query:>4} {g.n_init_labeled:>5} {g.n_round:>3} {job.seed:>4} {est}"
        )
    lines.append("")
    lines.append(render_totals(jobs, estimates))
    return "\n".join(lines)


def _totals_line(label: str, width: int, jobs: Sequence[Job], estimates: dict[str, float]) -> str:
    line = f"  {label:<{width}} {len(jobs):>4} runs ({len({j.group.id for j in jobs})} run groups)"
    if estimates:
        hours, unknown = estimated_hours(jobs, estimates)
        line += f"  est {hours:6.1f} GPU h" + (f" (+{unknown} runs without estimate)" if unknown else "")
    return line


def render_totals(jobs: Sequence[Job], estimates: dict[str, float]) -> str:
    """Runs, groups and (with `estimates`) estimated GPU hours per part, per GPU part and in total."""
    lines: list[str] = []
    width = max(len(name) for name in (*PARTS, *GPU_PARTS, "TOTAL"))
    for part in PARTS:
        lines.append(_totals_line(part, width, [j for j in jobs if j.group.part == part], estimates))
    if any(j.group.gpu_part for j in jobs):
        lines.append("  GPU parts (ADR 0016):")
        for gpu_part in GPU_PARTS:
            lines.append(_totals_line(gpu_part, width, [j for j in jobs if j.group.gpu_part == gpu_part], estimates))
    lines.append(_totals_line("TOTAL", width, jobs, estimates))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dalmax.campaign", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", default=None,
                       help=f"manifest path, or a comma-separated list merged into one plan (default {DEFAULT_MANIFEST})")
        p.add_argument("--micro", action="store_true", help=f"use {DEFAULT_MICRO_MANIFEST} (CPU smoke)")
        p.add_argument("--part", default="all",
                       help=f"{','.join(PARTS)} (comma-separated) or all; or {' / '.join(GPU_PARTS)}")
        p.add_argument("--seeds", default=None, help="comma-separated subset of the seeds (smoke); default all")
        p.add_argument(
            "--exclude-strategy", default=None,
            help="comma-separated strategy names to drop (e.g. AdversarialBIM,AdversarialDeepFool for CPU smoke)",
        )

    common(sub.add_parser("list", help="print the job table, counts and estimated GPU hours"))
    run_p = sub.add_parser("run", help="execute jobs sequentially via tools/trainer.py")
    common(run_p)
    run_p.add_argument("--dry-run", action="store_true", help="echo commands, execute nothing")
    run_p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="skip jobs whose leaf is complete (all artifacts present; default on)",
    )
    run_p.add_argument("--device", default=None, choices=["cuda", "cpu", "auto"],
                       help="default: cuda (cpu with --micro)")
    common(sub.add_parser("verify", help="OK/INCOMPLETE/MISSING per job + seed-consistency audit"))
    return parser


def verify_exit_code(statuses: Sequence[JobStatus], audits: Sequence[SeedAudit]) -> int:
    """0: everything OK; 1: an incomplete/missing job or a FAILed audit; 2: only audit WARNs."""
    if any(s.status != "OK" for s in statuses) or any(a.status in ("FAIL", "NO_DATA") for a in audits):
        return 1
    return 2 if any(a.status == "WARN" for a in audits) else 0


def _dispatch(args: argparse.Namespace) -> int:
    try:
        spec = args.manifest or (DEFAULT_MICRO_MANIFEST if args.micro else DEFAULT_MANIFEST)
        manifests = [load_manifest(path.strip()) for path in spec.split(",") if path.strip()]
        if not manifests:
            raise CampaignError("empty --manifest")
        selection = parse_selection(args.part)
        excluded = parse_exclude_strategies(args.exclude_strategy, manifests)
        estimates = {id(m): load_estimates(m) for m in manifests}
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    plan = plan_jobs(manifests, selection, seeds, excluded)
    if args.command == "list":
        if excluded:
            print(f"excluded strategies: {', '.join(sorted(excluded))}")
        for manifest in manifests:
            jobs = [j for seg in plan if seg.manifest is manifest for j in seg.jobs]
            print(f"manifest: {manifest.path}  results_root: {manifest.results_root}")
            print(render_job_table(jobs, manifest.primary_nq, estimates[id(manifest)]))
            print()
        if len(manifests) > 1:
            print(f"Execution plan ({len(plan)} segments):")
            for n, seg in enumerate(plan, start=1):
                print(f"  {n}. {Path(seg.manifest.path).name}: {len(seg.jobs)} runs")
            combined: dict[str, float] = {}
            for manifest in manifests:
                combined.update(estimates[id(manifest)])
            print("Combined:")
            print(render_totals([j for seg in plan for j in seg.jobs], combined))
        return 0
    if args.command == "verify":
        statuses = [verify_job(j, required_extra=seg.manifest.required_leaf_files)
                    for seg in plan for j in seg.jobs]
        print(render_verify(statuses))
        audits = audit_seed_consistency([j for seg in plan for j in seg.jobs])
        print()
        print(render_seed_audit(audits))
        return verify_exit_code(statuses, audits)

    device = args.device or ("cpu" if args.micro else "cuda")
    failed = 0
    for seg in plan:
        summary = run_jobs(
            seg.jobs, seg.manifest, device=device, skip_existing=args.skip_existing, dry_run=args.dry_run
        )
        failed += summary.failed
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        code = _dispatch(args)
        sys.stdout.flush()  # a closed pipe can also surface here, not only inside print()
        return code
    except BrokenPipeError:  # e.g. `campaign list | head`: exit quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
