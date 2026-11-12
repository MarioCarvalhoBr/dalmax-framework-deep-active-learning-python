"""Campaign manifests: load, validate, expand to jobs, run, verify.

`files_config/campaign/manifest.json` is the single source of truth for the
clean re-execution of everything the three papers need on
`DATA/daninhas_full_fixed` (see `.specs/experiments/campaign.md`, ADR 0008 and
ADR 0016); `manifest_kmh_sweep.json` is the paper-1 KMH sweep (ADR 0012). Each
*run group* is one distinct computation (params JSON + strategy + n_query +
n_init_labeled + n_round) swept over the manifest's seeds; a table row of any
paper that needs the same computation points to that one group (directly, or
through an *alias* entry that documents why two differently-spelled configs are
the same computation). Inside one manifest nothing runs twice
(`validate_manifest`). Across the main manifest and the KMH sweep, three groups
repeat the reference KMH configuration ON PURPOSE (`shared/kmh_nq100`,
`p1/KMH/nq50`, `p1/KMH/nq10` vs the sweep's `kmh/nq{100,50,10}/hier_L3_ref`, so
every sweep table is measured inside one session): `verify` over several
manifests reports whether each such pair selected the same images
(`cross_manifest_duplicates`, a free determinism check).

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
  `<results_root>/failures.log` of the job's manifest, continuing after a failure
  EXCEPT a lack of disk space: `run` refuses to start (exit 4) when the free space
  of a results file system is below the remaining runs x `RUN_DISK_BYTES` x 1.2
  (`dalmax/experiment/disk_space.py`), and stops the batch (exit 4) when a run
  exits with `NO_SPACE_EXIT_CODE` or the free space drops below one run.
- `space`: print that free-space check for the selection (exit 0 enough, 4 short).
- `verify [--part ...] [--micro]`: per job OK / INCOMPLETE / MISSING, reusing
  `dalmax.reporting.leaf_check.check_leaf` (plus the manifest's
  `required_leaf_files`), the seed-consistency audit (round-0 image paths from
  `round_queries.json`, pool indices from the log as fallback) and the data-order
  audit (`run_metadata.json` `data_order` hashes, KI-47) over every selected job;
  with several manifests also the duplicate-pair report (informational). A leaf with
  an integrity problem (truncated/empty JSON, wrong round count, empty checkpoint)
  is INCOMPLETE and re-run on resume; leaves are looked up only under the dataset
  folder of the group's params (`basename(data_dir)`), and a leaf of the plan under
  another dataset folder FAILs the data-order audit. The code-version audit prints
  the distinct `git_commit` values of the OK leaves (more than one, or a missing
  one, is a WARN that does not change the exit code; also in `pilot-report`).
  Exit code: 0 all OK; 1 an incomplete/missing job or a FAIL audit; 2 only audit
  WARNs (a run whose initial set or data order cannot be read).
- `pilot-report`: read-only summary of the pilot leaves (wall time vs estimate,
  `strategy_diagnostics`, degeneracy flags; `dalmax/reporting/pilot_report.py`).
- Every subcommand accepts `--exclude-strategy A,B` (e.g. the two adversarial
  baselines, far too slow for a CPU smoke run): jobs of those strategies are
  dropped from the list/run/verify/audit. The manifest still contains them.
- Every subcommand accepts `--only-group ID[,ID]` (restrict to those groups) and
  `--pilot` (= the pilot groups of the selection, `pilot_group_ids`, seed 1 unless
  `--seeds` says otherwise): seed 1 of the VAAL and WAAL groups at the largest
  budget of the selection plus RandomSampling at that budget. A later full run
  skips the completed pilot leaves (skip-existing).

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
from dalmax.data_order import relative_image_paths
from dalmax.experiment.disk_space import (
    DISK_MARGIN,
    NO_SPACE_EXIT_CODE,
    RUN_DISK_BYTES,
    file_system_id,
    format_gb,
    free_bytes,
    required_bytes,
)
from dalmax.query_strategies.registry import (
    GENERIC_REPRESENTATION_NAME,
    REPRESENTATION_PRESET_Q,
    REPRESENTATION_PRESETS,
)
from dalmax.reporting.leaf_check import ROUND_HISTORY_FILES, ROUND_QUERIES_FILE, check_leaf
from dalmax.reporting.pilot_report import PilotRow, read_campaign_log_elapsed, render_pilot_report
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
    # `basename(data_dir)` of the group's params (`params_dataset_folder`), the folder the trainer writes the
    # leaves under; set by `load_manifest`. Leaves are looked up only there (`find_leaves`).
    dataset_folder: str = ""


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


def params_dataset_folder(params_json: str, dataset_name: str) -> str:
    """The dataset folder a run of `params_json` writes its leaf under: `basename(data_dir)` of the
    params' `dataset_name` block, the derivation of `dalmax.experiment.runner` and the loaders."""
    path = _resolve_path(params_json)
    try:
        data_dir = json.loads(path.read_text())[dataset_name]["data_dir"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CampaignError(f"cannot read {dataset_name}.data_dir of params {path}: {exc!r}") from exc
    folder = os.path.basename(str(data_dir).rstrip("/"))
    if not folder:
        raise CampaignError(f"params {path}: {dataset_name}.data_dir {data_dir!r} names no folder")
    return folder


def _to_group(raw: dict[str, Any], default_seeds: tuple[int, ...], dataset_name: str) -> RunGroup:
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
            dataset_folder=params_dataset_folder(raw["params_json"], dataset_name),
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
    dataset_name = raw.get("dataset_name", "DANINHAS")
    manifest = Manifest(
        name=raw.get("name", p.stem),
        dataset_name=dataset_name,
        results_root=raw["results_root"],
        seeds=seeds,
        primary_nq=int(raw.get("primary_nq", 100)),
        groups=tuple(_to_group(g, seeds, dataset_name) for g in raw["groups"]),
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


def selected_groups(
    manifests: Sequence[Manifest], selection: Selection, exclude_strategies: Collection[str] = ()
) -> list[RunGroup]:
    """Every group of `manifests` the selection keeps (in manifest order)."""
    return [g for m in manifests for g in m.groups
            if selection.selects(g) and g.strategy_name not in exclude_strategies]


def restrict_plan(plan: Sequence[Segment], group_ids: Collection[str], available: Collection[str]) -> list[Segment]:
    """Keep only the jobs of `group_ids` (order preserved, empty segments dropped).

    Fails fast on an id that is not among `available` (the groups of the selection),
    e.g. a typo or a group of the other GPU part.
    """
    unknown = sorted(set(group_ids) - set(available))
    if unknown:
        raise CampaignError(f"--only-group: {unknown} not in the selected groups (check --part / --manifest)")
    wanted = set(group_ids)
    kept = [Segment(seg.manifest, tuple(j for j in seg.jobs if j.group.id in wanted)) for seg in plan]
    return [seg for seg in kept if seg.jobs]


# The pilot (2026-11-12): the two long DeepAL+ baselines whose costs are FLOP estimates (VAAL, WAAL) and one
# cheap strategy, at the largest budget of the selection, seed 1 only, run before a GPU part's full plan.
PILOT_STRATEGIES: tuple[str, ...] = ("VAAL", "WAAL")
PILOT_CHEAP_STRATEGY = "RandomSampling"


def pilot_group_ids(groups: Sequence[RunGroup]) -> tuple[str, ...]:
    """The pilot of a selection: its VAAL and WAAL groups at the largest `n_query` they have in the
    selection (nq100 in `part_1_gpu`, nq50 in `part_2_gpu`), plus `RandomSampling` at that budget."""
    long = [g for g in groups if g.strategy_name in PILOT_STRATEGIES]
    if not long:
        raise CampaignError(f"--pilot: the selection has no {'/'.join(PILOT_STRATEGIES)} group")
    budget = max(g.n_query for g in long)
    picked = [g for g in long if g.n_query == budget]
    missing = sorted(set(PILOT_STRATEGIES) - {g.strategy_name for g in picked})
    if missing:
        raise CampaignError(f"--pilot: no {missing} group at n_query={budget} in the selection")
    cheap = [g for g in groups if g.strategy_name == PILOT_CHEAP_STRATEGY and g.n_query == budget]
    if not cheap:
        raise CampaignError(f"--pilot: no {PILOT_CHEAP_STRATEGY} group at n_query={budget} in the selection")
    return tuple(g.id for g in (cheap[0], *picked))


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


def _own_dataset_folder(group: RunGroup) -> str:
    if not group.dataset_folder:
        raise CampaignError(f"run group {group.id!r} has no dataset_folder (build it with load_manifest, or set "
                            "dataset_folder=params_dataset_folder(params_json, dataset_name))")
    return group.dataset_folder


def leaf_glob(group: RunGroup, seed: int, dataset_folder: str | None = None) -> str:
    """Glob (relative to the group's `dir_results`) matching this job's leaf dir under `dataset_folder`
    (default: the group's own, from its params' `data_dir`; `"*"` = any folder) and any n_epoch."""
    folder = dataset_folder if dataset_folder is not None else _own_dataset_folder(group)
    return (f"{folder}/SEED_{seed}/NQ_{group.n_query}_NIL_{group.n_init_labeled}_NR_{group.n_round}_NE_*/"
            f"{group.strategy_name}")


def _leaves(group: RunGroup, seed: int, dataset_folder: str, base: Path | None) -> list[Path]:
    root = (base if base is not None else REPO_ROOT) / group.dir_results
    if not root.is_dir():
        return []
    return sorted(root.glob(leaf_glob(group, seed, dataset_folder)))


def find_leaves(group: RunGroup, seed: int, *, base: Path | None = None) -> list[Path]:
    """Every existing leaf directory of a job under the group's OWN dataset folder (any n_epoch), sorted.

    A leaf under another dataset folder (e.g. `daninhas_full`, which sorts before
    `daninhas_full_fixed`) is never returned: it is another computation
    (`find_foreign_leaves`, a data-order audit FAIL). `base` overrides the
    directory `dir_results` is resolved against (default the repo root); used by
    the report/tests to point at another tree.
    """
    return _leaves(group, seed, _own_dataset_folder(group), base)


def find_foreign_leaves(group: RunGroup, seed: int, *, base: Path | None = None) -> list[Path]:
    """Leaves of this job under a dataset folder that is NOT the group's own, sorted."""
    own = _own_dataset_folder(group)
    return [leaf for leaf in _leaves(group, seed, "*", base) if leaf_dataset_folder(leaf) != own]


def leaf_dataset_folder(leaf: Path) -> str:
    """`<dataset>` of `<dir_results>/<dataset>/SEED_s/NQ_.../<strategy>`."""
    return leaf.parent.parent.parent.name


def find_leaf(group: RunGroup, seed: int, *, base: Path | None = None) -> Path | None:
    """The first existing leaf directory of a job (see `find_leaves`), or None."""
    matches = find_leaves(group, seed, base=base)
    return matches[0] if matches else None


def job_done(job: Job, required_extra: tuple[str, ...] = ()) -> bool:
    """True iff the job's leaf carries the FULL, intact artifact set (`check_leaf(...).complete`,
    plus the manifest's `required_leaf_files` passed as `required_extra`), so a run killed
    mid-report or a leaf with a truncated/empty file is re-run on resume (into the same leaf,
    rewriting its files) instead of being skipped. (`results.json` is written last by the reporter.)"""
    leaf = find_leaf(job.group, job.seed)
    if leaf is None:
        return False
    return check_leaf(leaf, required_extra).complete


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
    stopped_no_space: bool = False  # the batch was stopped for lack of disk space (exit 4)


# `campaign run`/`space` exit code when the free space of a results file system is short.
EXIT_NO_SPACE = 4


def run_jobs(
    jobs: Sequence[Job],
    manifest: Manifest,
    *,
    device: str,
    skip_existing: bool = True,
    dry_run: bool = False,
    executor: Callable[[list[str], dict[str, str]], int] = _default_executor,
    echo: Callable[[str], None] = print,
    free_space: Callable[[Path], int] | None = None,
    run_bytes: int = RUN_DISK_BYTES,
) -> RunSummary:
    """Execute jobs (all of `manifest`) sequentially; a failing job is logged and the batch continues,
    except for a lack of disk space: before each run the results file system must hold one more run
    (`required_bytes(1)`), and a run that exits with `NO_SPACE_EXIT_CODE` stops the batch too
    (`summary.stopped_no_space`), instead of training the next runs for nothing. `free_space`
    defaults to `free_bytes` (injectable for tests)."""
    free_space = free_space or free_bytes
    root = REPO_ROOT / manifest.results_root
    campaign_log = root / "campaign.log"
    failures_log = root / "failures.log"
    summary = RunSummary()
    env = build_env(device)
    total = len(jobs)
    one_run = required_bytes(1, run_bytes=run_bytes)

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
        free = free_space(root)
        if free < one_run:
            summary.stopped_no_space = True
            log(f"{head} STOP no space: {format_gb(free)} free on the results file system, "
                f"{format_gb(one_run)} needed for one run; free space, then re-run (resumes)")
            break
        log(f"{head} START")
        t0 = time.time()
        code = executor(cmd, env)
        elapsed = time.time() - t0
        if code != 0:
            summary.failed += 1
            log(f"{head} FAILED exit={code} elapsed={elapsed:.0f}s")
            _append(failures_log, f"{_stamp()} FAILED {job.label()} exit={code}")
            if code == NO_SPACE_EXIT_CODE:
                summary.stopped_no_space = True
                log(f"{head} STOP no space: the run failed for lack of disk space (exit {code}); "
                    "free space, then re-run (resumes)")
                break
        else:
            summary.executed += 1
            log(f"{head} DONE elapsed={elapsed:.0f}s")
    log(
        f"CAMPAIGN END executed={summary.executed} skipped={summary.skipped} "
        f"failed={summary.failed} dry_run={summary.dry}"
        + (" STOPPED=no_space" if summary.stopped_no_space else "")
    )
    return summary


@dataclass(frozen=True)
class SpaceCheck:
    """Free space of one file system against the runs still to execute into the results roots on it."""

    roots: tuple[str, ...]
    remaining_runs: int
    needed: int
    free: int

    @property
    def ok(self) -> bool:
        return self.free >= self.needed


def space_checks(
    plan: Sequence[Segment],
    *,
    skip_existing: bool = True,
    free_space: Callable[[Path], int] | None = None,
    device_of: Callable[[Path], int] | None = None,
    run_bytes: int = RUN_DISK_BYTES,
) -> list[SpaceCheck]:
    """One `SpaceCheck` per file system holding a results root of `plan`. Remaining runs = the
    jobs without a complete leaf (all jobs with `skip_existing=False`); needed =
    `required_bytes(remaining)` (runs x per-run size x 20% margin). `free_space` / `device_of`
    default to `free_bytes` / `file_system_id` (injectable for tests)."""
    free_space = free_space or free_bytes
    device_of = device_of or file_system_id
    remaining: dict[str, int] = {}
    for seg in plan:
        todo = sum(1 for j in seg.jobs if not (skip_existing and job_done(j, seg.manifest.required_leaf_files)))
        remaining[seg.manifest.results_root] = remaining.get(seg.manifest.results_root, 0) + todo
    by_device: dict[int, list[str]] = {}
    for root in remaining:
        by_device.setdefault(device_of(REPO_ROOT / root), []).append(root)
    checks = []
    for roots in by_device.values():
        runs = sum(remaining[r] for r in roots)
        checks.append(SpaceCheck(tuple(roots), runs, required_bytes(runs, run_bytes=run_bytes),
                                 free_space(REPO_ROOT / roots[0])))
    return checks


def render_space_checks(checks: Sequence[SpaceCheck], run_bytes: int = RUN_DISK_BYTES) -> str:
    lines = [f"Free-space check (results file systems; remaining runs x per-run size x {DISK_MARGIN}):"]
    for c in checks:
        verdict = "OK" if c.ok else "NOT ENOUGH"
        lines.append(
            f"  {verdict}: {', '.join(c.roots)}: {c.remaining_runs} remaining runs x {run_bytes / 1e6:.0f} MB x "
            f"{DISK_MARGIN} = {format_gb(c.needed)} needed, {format_gb(c.free)} free"
        )
    return "\n".join(lines)


# --- verify -------------------------------------------------------------------


@dataclass(frozen=True)
class JobStatus:
    job: Job
    status: str  # OK | INCOMPLETE | MISSING
    missing_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()  # integrity problems (`LeafCheck.problems`): they make the job INCOMPLETE
    leaf: Path | None = None


def verify_job(job: Job, *, base: Path | None = None, required_extra: tuple[str, ...] = ()) -> JobStatus:
    leaf = find_leaf(job.group, job.seed, base=base)
    if leaf is None:
        return JobStatus(job, "MISSING")
    check = check_leaf(leaf, required_extra)
    status = "OK" if check.complete else "INCOMPLETE"
    return JobStatus(job, status, check.missing, check.warnings, check.problems, leaf)


# --- code-version audit (git_commit of every OK leaf) ---------------------------


@dataclass(frozen=True)
class CommitAudit:
    """Which code versions (`run_metadata.json` `git_commit`) produced the leaves."""

    status: str  # PASS | WARN | NO_DATA
    counts: tuple[tuple[str, int, str], ...] = ()  # (commit, number of leaves, first job label)
    unknown: tuple[str, ...] = ()  # job labels whose git_commit is missing / None / unreadable


def read_git_commit(leaf: Path) -> str | None:
    """`git_commit` of `<leaf>/run_metadata.json`, or None (missing, null, unreadable)."""
    try:
        commit = json.loads((leaf / "run_metadata.json").read_text()).get("git_commit")
    except (OSError, ValueError, AttributeError):
        return None
    return str(commit) if commit else None


def audit_git_commits(leaves: Sequence[tuple[str, Path]]) -> CommitAudit:
    """Distinct `git_commit` values over `(job label, leaf)` pairs. More than one value, or a leaf
    without one, is a WARN (mixed or unknown code versions); it never changes `verify`'s exit code."""
    by_commit: dict[str, list[str]] = {}
    unknown: list[str] = []
    for label, leaf in leaves:
        commit = read_git_commit(leaf)
        if commit is None:
            unknown.append(label)
        else:
            by_commit.setdefault(commit, []).append(label)
    if not leaves:
        return CommitAudit("NO_DATA")
    counts = tuple(sorted(((c, len(ls), ls[0]) for c, ls in by_commit.items()), key=lambda t: (-t[1], t[0])))
    status = "WARN" if len(counts) > 1 or unknown else "PASS"
    return CommitAudit(status, counts, tuple(unknown))


def render_commit_audit(audit: CommitAudit, max_listed: int = 5) -> str:
    lines = ["Code-version audit (run_metadata.json git_commit of every OK leaf):"]
    if audit.status == "NO_DATA":
        lines.append("  no OK leaves")
        return "\n".join(lines)
    n_leaves = sum(n for _, n, _ in audit.counts) + len(audit.unknown)
    lines.append(f"  {audit.status} ({n_leaves} leaves, {len(audit.counts)} distinct commit(s))")
    for commit, count, example in audit.counts:
        lines.append(f"    {commit}: {count} leaves, e.g. {example}")
    if audit.unknown:
        shown = ", ".join(audit.unknown[:max_listed]) + (" ..." if len(audit.unknown) > max_listed else "")
        lines.append(f"    no git_commit ({len(audit.unknown)} leaves): {shown}")
    if len(audit.counts) > 1:
        lines.append("  WARN: the leaves were produced by more than one code version (mixed commits)")
    if audit.unknown:
        lines.append("  WARN: some leaves do not record the code version (git_commit missing or null)")
    return "\n".join(lines)


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
    basis: str = "pool indices"  # "image paths", "pool indices" or "image paths + pool indices"


@dataclass(frozen=True)
class InitialSet:
    """The round-0 labeled set of one run: pool indices and, when the leaf has
    `round_queries.json`, the sorted relative image paths (KI-47: indices alone
    name images only under one data order)."""

    idxs: list[int] | None
    paths: list[str] | None = None

    @property
    def readable(self) -> bool:
        return self.idxs is not None or self.paths is not None


def read_initial_idxs(leaf: Path) -> list[int] | None:
    """Parse `Initial labeled idxs (sorted): [...]` from `<leaf>/log-dalmax.log`."""
    log = leaf / "log-dalmax.log"
    if not log.is_file():
        return None
    match = INITIAL_IDXS_RE.search(log.read_text(errors="replace"))
    return json.loads(match.group(1)) if match else None


def read_round_paths(leaf: Path) -> list[list[str]] | None:
    """Per round, the sorted relative paths (`<class>/<file>`) labeled in it, from
    `round_queries.json`; None when the file is absent or has no paths."""
    path = leaf / ROUND_QUERIES_FILE
    if not path.is_file():
        return None
    try:
        rounds = sorted(json.loads(path.read_text())["rounds"], key=lambda r: int(r["round"]))
        return [sorted(relative_image_paths(r["paths"])) for r in rounds]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_initial_set(leaf: Path) -> InitialSet:
    """Round-0 set of a leaf: indices and image paths from `round_queries.json` when it
    has them, else the indices of the log line (trees older than ADR 0016)."""
    path = leaf / ROUND_QUERIES_FILE
    if path.is_file():
        try:
            rounds = json.loads(path.read_text())["rounds"]
            initial = next(r for r in rounds if int(r["round"]) == 0)
            return InitialSet([int(i) for i in initial["pool_idxs"]], sorted(relative_image_paths(initial["paths"])))
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            pass  # malformed/older format: fall back to the log
    return InitialSet(read_initial_idxs(leaf))


def audit_seed_consistency(jobs: Sequence[Job], *, base: Path | None = None) -> list[SeedAudit]:
    """For each seed, every run (all strategies/configs, `FullSupervised`
    excluded) must have started from the IDENTICAL initial labeled set -- the
    fair-comparison premise of the whole campaign. Compared by image paths
    (`round_queries.json`) wherever available, so two runs that start from the
    same indices but different images FAIL (KI-47), and by pool indices
    (log line) as the fallback for older leaves; any difference is a FAIL."""
    by_seed: dict[int, list[tuple[str, InitialSet]]] = {}
    for job in jobs:
        if job.group.strategy_name in AUDIT_EXCLUDED_STRATEGIES:
            continue
        leaf = find_leaf(job.group, job.seed, base=base)
        if leaf is None:
            continue  # not run yet: `verify` reports it as MISSING
        by_seed.setdefault(job.seed, []).append((job.group.id, read_initial_set(leaf)))

    seeds = sorted({j.seed for j in jobs if j.group.strategy_name not in AUDIT_EXCLUDED_STRATEGIES})
    audits: list[SeedAudit] = []
    for seed in seeds:
        runs = by_seed.get(seed, [])
        readable = [(rid, s) for rid, s in runs if s.readable]
        unreadable = tuple(rid for rid, s in runs if not s.readable)
        if not readable:
            # Runs exist but none can be compared (missing/unreadable log): WARN, not silence.
            status = "WARN" if unreadable else "NO_DATA"
            audits.append(SeedAudit(seed, status, len(runs), unreadable=unreadable))
            continue
        ref_idxs = next((s.idxs for _, s in readable if s.idxs is not None), None)
        ref_paths = next((s.paths for _, s in readable if s.paths is not None), None)
        offenders = tuple(
            rid for rid, s in readable
            if (s.idxs is not None and s.idxs != ref_idxs) or (s.paths is not None and s.paths != ref_paths)
        )
        with_paths = sum(s.paths is not None for _, s in readable)
        basis = ("image paths" if with_paths == len(readable)
                 else "pool indices" if with_paths == 0 else "image paths + pool indices")
        status = "FAIL" if offenders else ("WARN" if unreadable else "PASS")
        audits.append(SeedAudit(seed, status, len(readable), readable[0][0], offenders, unreadable, basis))
    return audits


def render_seed_audit(audits: Sequence[SeedAudit]) -> str:
    lines = ["Seed-consistency audit (initial labeled set identical across all strategies/configs):"]
    for a in audits:
        line = f"  seed {a.seed}: {a.status} ({a.n_runs} runs compared"
        line += f" by {a.basis}, reference {a.reference})" if a.reference else ")"
        lines.append(line)
        for rid in a.offenders:
            lines.append(f"    OFFENDER (different initial set): {rid}")
        for rid in a.unreadable:
            lines.append(f"    unreadable round_queries.json and log-dalmax.log / no initial-idxs line: {rid}")
    return "\n".join(lines)


# --- data-order audit (KI-47) -------------------------------------------------


@dataclass(frozen=True)
class DataOrderAudit:
    """Do all leaves of one dataset folder index the same ordered train and test image lists?"""

    dataset: str
    status: str  # PASS | FAIL | WARN
    n_runs: int
    # (train hash, test hash) -> (number of leaves, first job label), one entry per distinct order
    variants: tuple[tuple[str, str, int, str], ...] = ()
    missing: tuple[str, ...] = ()  # job labels whose run_metadata.json has no `data_order`
    # job labels with a leaf under THIS folder although their params' data_dir names another one
    foreign: tuple[str, ...] = ()


def read_data_order(leaf: Path) -> tuple[str, str] | None:
    """`(train_paths_hash, test_paths_hash)` of `run_metadata.json`'s `data_order`, or None."""
    try:
        order = json.loads((leaf / "run_metadata.json").read_text())["data_order"]
        return str(order["train_paths_hash"]), str(order["test_paths_hash"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def audit_data_order(jobs: Sequence[Job], *, base: Path | None = None) -> list[DataOrderAudit]:
    """Per dataset folder, every leaf (every strategy, the upper bound included) must
    record the same `data_order` hashes: otherwise the same index named different
    images in different runs (FAIL). Leaves without the record (written before
    KI-47) give WARN. A leaf of a job under a dataset folder other than the one of
    its params' `data_dir` (e.g. a `daninhas_full` leaf next to the
    `daninhas_full_fixed` ones, never picked by `find_leaf`) is a FAIL of that folder:
    the results root mixes leaves of more than one dataset."""
    by_dataset: dict[str, list[tuple[str, tuple[str, str] | None]]] = {}
    foreign: dict[str, list[str]] = {}
    for job in jobs:
        for leaf in find_foreign_leaves(job.group, job.seed, base=base):
            foreign.setdefault(leaf_dataset_folder(leaf), []).append(job.label())
        leaf = find_leaf(job.group, job.seed, base=base)
        if leaf is None:
            continue
        by_dataset.setdefault(leaf_dataset_folder(leaf), []).append((job.label(), read_data_order(leaf)))
    audits: list[DataOrderAudit] = []
    for dataset, labels in sorted(foreign.items()):
        audits.append(DataOrderAudit(dataset, "FAIL", len(labels), foreign=tuple(labels)))
    for dataset, runs in sorted(by_dataset.items()):
        counts: dict[tuple[str, str], list[str]] = {}
        for label, order in runs:
            if order is not None:
                counts.setdefault(order, []).append(label)
        variants = tuple((t, s, len(labels), labels[0]) for (t, s), labels in counts.items())
        missing = tuple(label for label, order in runs if order is None)
        status = "FAIL" if len(variants) > 1 else ("WARN" if missing else "PASS")
        audits.append(DataOrderAudit(dataset, status, len(runs), variants, missing))
    return audits


def render_data_order_audit(audits: Sequence[DataOrderAudit], max_listed: int = 5) -> str:
    lines = ["Data-order audit (every leaf of a dataset indexes the same ordered image lists, KI-47):"]
    if not audits:
        lines.append("  no leaves")
    for a in audits:
        if a.foreign:
            shown = ", ".join(a.foreign[:max_listed]) + (" ..." if len(a.foreign) > max_listed else "")
            lines.append(f"  {a.dataset}: {a.status} ({len(a.foreign)} leaves of the plan under a dataset folder that "
                         f"is not their params' data_dir: the results root mixes datasets): {shown}")
            continue
        lines.append(f"  {a.dataset}: {a.status} ({a.n_runs} runs, {len(a.variants)} distinct order(s))")
        for train, test, count, example in a.variants:
            lines.append(f"    train {train} / test {test}: {count} runs, e.g. {example}")
        if a.missing:
            shown = ", ".join(a.missing[:max_listed]) + (" ..." if len(a.missing) > max_listed else "")
            lines.append(f"    no data_order in run_metadata.json ({len(a.missing)} runs): {shown}")
    return "\n".join(lines)


# --- intentional duplicate computations across manifests (informational) -------


@dataclass(frozen=True)
class DuplicatePairCheck:
    """Did two groups with the same effective signature (in different manifests) select the same images?"""

    first: str
    second: str
    seed: int
    status: str  # SAME | DIFFERENT | N/A
    detail: str = ""


def cross_manifest_duplicates(manifests: Sequence[Manifest]) -> list[tuple[RunGroup, RunGroup]]:
    """Pairs of groups of DIFFERENT manifests that are the same computation (`group_signature`).
    Within a manifest `validate_manifest` already forbids any."""
    seen: dict[str, tuple[int, RunGroup]] = {}
    pairs: list[tuple[RunGroup, RunGroup]] = []
    for mi, manifest in enumerate(manifests):
        if manifest.dataset_name != manifests[0].dataset_name:
            continue
        for group in manifest.groups:
            sig = group_signature(manifest, group)
            other = seen.get(sig)
            if other is None:
                seen[sig] = (mi, group)
            elif other[0] != mi:
                pairs.append((other[1], group))
    return pairs


def check_duplicate_pairs(
    pairs: Sequence[tuple[RunGroup, RunGroup]], seeds: Collection[int], *, base: Path | None = None
) -> list[DuplicatePairCheck]:
    """For each pair and seed: SAME if both leaves labeled the same images in every round
    (`round_queries.json`), DIFFERENT (first differing round named), N/A if a side is missing."""
    checks: list[DuplicatePairCheck] = []
    for a, b in pairs:
        for seed in sorted(set(a.seeds) & set(b.seeds) & set(seeds)):
            leaves = [find_leaf(g, seed, base=base) for g in (a, b)]
            rounds = [read_round_paths(leaf) if leaf is not None else None for leaf in leaves]
            if rounds[0] is None or rounds[1] is None:
                checks.append(DuplicatePairCheck(a.id, b.id, seed, "N/A", "a leaf or its round_queries.json is missing"))
                continue
            if rounds[0] == rounds[1]:
                checks.append(DuplicatePairCheck(a.id, b.id, seed, "SAME", f"{len(rounds[0])} rounds"))
                continue
            first = next((r for r, (x, y) in enumerate(zip(*rounds, strict=False)) if x != y), min(map(len, rounds)))
            checks.append(DuplicatePairCheck(a.id, b.id, seed, "DIFFERENT", f"first differing round {first}"))
    return checks


def render_duplicate_pairs(checks: Sequence[DuplicatePairCheck]) -> str:
    lines = ["Intentional duplicate computations across manifests (informational; same seed + config "
             "should select the same images):"]
    if not checks:
        lines.append("  no pair inside this selection")
    for c in checks:
        lines.append(f"  {c.status:<9} {c.first} == {c.second} seed={c.seed} ({c.detail})")
    return "\n".join(lines)


def render_verify(statuses: Sequence[JobStatus]) -> str:
    lines: list[str] = []
    for st in statuses:
        extra = ""
        if st.missing_files:
            extra = f" (missing {', '.join(st.missing_files)})"
        if st.problems:
            extra += f" (integrity: {'; '.join(st.problems)})"
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
        only = p.add_mutually_exclusive_group()
        only.add_argument("--only-group", default=None,
                          help="comma-separated group ids: restrict the selection to them (fail fast on unknown ids)")
        only.add_argument("--pilot", action="store_true",
                          help=f"only the pilot groups of the selection ({PILOT_CHEAP_STRATEGY} + "
                               f"{'/'.join(PILOT_STRATEGIES)} at its largest budget), seed 1 unless --seeds is given")

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
    common(sub.add_parser("verify", help="OK/INCOMPLETE/MISSING per job + seed-consistency and data-order audits"))
    common(sub.add_parser("space", help="free space of the results file systems vs the remaining runs (exit 4 short)"))
    common(sub.add_parser("pilot-report", help="read-only: wall time vs estimate and strategy_diagnostics of the "
                                               "pilot leaves (implies --pilot)"))
    return parser


def verify_exit_code(
    statuses: Sequence[JobStatus],
    audits: Sequence[SeedAudit],
    data_order_audits: Sequence[DataOrderAudit] = (),
) -> int:
    """0: everything OK; 1: an incomplete/missing job (integrity problems included) or a FAILed audit
    (a data-order FAIL includes leaves under a foreign dataset folder); 2: only seed/data-order audit
    WARNs. The code-version audit (`audit_git_commits`) is informational and never changes it."""
    if any(s.status != "OK" for s in statuses) or any(a.status in ("FAIL", "NO_DATA") for a in audits):
        return 1
    if any(a.status == "FAIL" for a in data_order_audits):
        return 1
    warn = any(a.status == "WARN" for a in audits) or any(a.status == "WARN" for a in data_order_audits)
    return 2 if warn else 0


def _pilot_rows(plan: Sequence[Segment], estimates: dict[int, dict[str, float]],
                all_groups: Sequence[RunGroup]) -> list[PilotRow]:
    """One `PilotRow` per pilot job (leaf, wall time from the manifest's campaign.log, estimate,
    number of runs of the same strategy in the whole selection for the projection)."""
    rows = []
    for seg in plan:
        log = REPO_ROOT / seg.manifest.results_root / "campaign.log"
        for job in seg.jobs:
            runs_in_selection = sum(len(g.seeds) for g in all_groups if g.strategy_name == job.group.strategy_name)
            rows.append(PilotRow(
                label=job.label(),
                strategy=job.group.strategy_name,
                leaf=find_leaf(job.group, job.seed),
                campaign_log_seconds=read_campaign_log_elapsed(log, job.group.id, job.seed),
                estimate_seconds=estimates[id(seg.manifest)].get(job.group.id),
                runs_in_selection=runs_in_selection,
            ))
    return rows


def _dispatch(args: argparse.Namespace) -> int:
    try:
        spec = args.manifest or (DEFAULT_MICRO_MANIFEST if args.micro else DEFAULT_MANIFEST)
        manifests = [load_manifest(path.strip()) for path in spec.split(",") if path.strip()]
        if not manifests:
            raise CampaignError("empty --manifest")
        selection = parse_selection(args.part)
        excluded = parse_exclude_strategies(args.exclude_strategy, manifests)
        estimates = {id(m): load_estimates(m) for m in manifests}
        groups = selected_groups(manifests, selection, excluded)
        pilot = args.pilot or args.command == "pilot-report"
        only = pilot_group_ids(groups) if pilot else (
            [s.strip() for s in args.only_group.split(",") if s.strip()] if args.only_group else None)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        if pilot and seeds is None:
            seeds = [min(manifests[0].seeds)]
        plan = plan_jobs(manifests, selection, seeds, excluded)
        if only is not None:
            plan = restrict_plan(plan, only, [g.id for g in groups])
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.command == "pilot-report":
        rows = _pilot_rows(plan, estimates, groups)
        print(render_pilot_report(rows))
        print()
        print(render_commit_audit(audit_git_commits([(r.label, r.leaf) for r in rows if r.leaf is not None])))
        return 0
    if args.command == "space":
        checks = space_checks(plan)
        print(render_space_checks(checks))
        return 0 if all(c.ok for c in checks) else EXIT_NO_SPACE
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
        jobs = [j for seg in plan for j in seg.jobs]
        statuses = [verify_job(j, required_extra=seg.manifest.required_leaf_files)
                    for seg in plan for j in seg.jobs]
        print(render_verify(statuses))
        print()
        print(render_commit_audit(audit_git_commits(
            [(s.job.label(), s.leaf) for s in statuses if s.status == "OK" and s.leaf is not None])))
        audits = audit_seed_consistency(jobs)
        print()
        print(render_seed_audit(audits))
        order_audits = audit_data_order(jobs)
        print()
        print(render_data_order_audit(order_audits))
        if len(manifests) > 1:
            in_plan = {j.group.id for j in jobs}
            pairs = [(a, b) for a, b in cross_manifest_duplicates(manifests) if a.id in in_plan and b.id in in_plan]
            print()
            print(render_duplicate_pairs(check_duplicate_pairs(pairs, {j.seed for j in jobs})))
        return verify_exit_code(statuses, audits, order_audits)

    device = args.device or ("cpu" if args.micro else "cuda")
    if not args.dry_run:
        checks = space_checks(plan, skip_existing=args.skip_existing)
        if not all(c.ok for c in checks):
            print(render_space_checks(checks), file=sys.stderr)
            print("ERROR: not enough free space for the remaining runs; nothing was started. Free space on the "
                  "results file system (the Drive on Colab) or run a smaller selection.", file=sys.stderr)
            return EXIT_NO_SPACE
    failed = 0
    for seg in plan:
        summary = run_jobs(
            seg.jobs, seg.manifest, device=device, skip_existing=args.skip_existing, dry_run=args.dry_run
        )
        failed += summary.failed
        if summary.stopped_no_space:
            print("ERROR: the batch was stopped for lack of disk space (see campaign.log); free space and "
                  "re-run the same command to resume.", file=sys.stderr)
            return EXIT_NO_SPACE
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
