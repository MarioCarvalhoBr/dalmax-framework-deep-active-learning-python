"""Single-campaign manifest: load, validate, expand to jobs, run, verify.

`files_config/campaign/manifest.json` is the single source of truth for the
one-shot Colab A100 re-execution of everything the three papers need (see
`.specs/experiments/campaign-a100.md` and ADR 0008). Each *run group* is one
distinct computation (params JSON + strategy + n_query + n_init_labeled +
n_round) swept over the manifest's seeds; a table row of any paper that needs
the same computation points to that one group (directly, or through an
*alias* entry that documents why two differently-spelled configs are the same
computation). Nothing runs twice.

CLI (`python -m dalmax.campaign <cmd>`):

- `list`: print the job table (counts per part and in total).
- `run --part {paper1,upper_bound,rnhal,texhal,all}[,...] [--micro] [--dry-run]
  [--skip-existing|--no-skip-existing] [--device cuda]`: execute the jobs
  sequentially through `trainer.py` (subprocess), logging to
  `<results_root>/campaign.log` and `<results_root>/failures.log`, continuing
  after a failure.
- `verify [--part ...] [--micro]`: per job OK / INCOMPLETE / MISSING, reusing
  `dalmax.reporting.results_doctor.check_leaf`.

Execution order (fixed, independent of `--part`): paper 1 at n_query=100
first (the shared canonical runs -- KMH and RandomSampling, which papers 2/3 reuse --
land early, so papers 2/3 comparisons can already be read), then rnhal, texhal,
then paper 1 at n_query=50 and 10, then the upper bound. If a session
budget runs out, what is missing is the least reusable part.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import FULL_SUPERVISED_STRATEGY
from dalmax.query_strategies.registry import (
    GENERIC_REPRESENTATION_NAME,
    REPRESENTATION_PRESET_Q,
    REPRESENTATION_PRESETS,
)
from dalmax.reporting.results_doctor import check_leaf
from dalmax.seeding import CUBLAS_WORKSPACE_CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = "files_config/campaign/manifest.json"
DEFAULT_MICRO_MANIFEST = "files_config/campaign/manifest_micro.json"

# Layout/ownership rule: a run group lives under the FIRST paper that consumes
# it (paper 1 < 2 < 3) and its `used_by` lists every consumer; e.g. paper 1's
# `KMH@100` is also the "w/o representation module" row of papers 2 and 3.
PARTS: tuple[str, ...] = ("paper1", "upper_bound", "rnhal", "texhal")

# Execution phases (see module docstring). Index = order of execution.
PHASE_LABELS: tuple[str, ...] = (
    "paper1_nq100",
    "rnhal",
    "texhal",
    "paper1_other_nq",
    "upper_bound",
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

    leaves: dict[tuple, str] = {}
    for group in manifest.groups:
        if group.part not in PARTS:
            raise CampaignError(f"{group.id}: part {group.part!r} not in {PARTS}")
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
        return 0 if group.n_query == primary_nq else 3
    return {"rnhal": 1, "texhal": 2, "upper_bound": 4}[group.part]


def parse_parts(spec: str) -> tuple[str, ...]:
    """`"all"` or a comma-separated subset of `PARTS`."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise CampaignError("empty --part")
    if "all" in names:
        return PARTS
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise CampaignError(f"unknown part(s) {bad}; valid: {', '.join(PARTS)} or 'all'")
    return tuple(n for n in PARTS if n in names)


def expand_jobs(
    manifest: Manifest, parts: Sequence[str] = PARTS, seeds: Sequence[int] | None = None
) -> list[Job]:
    """All `(group, seed)` jobs of `parts`, in execution order. `seeds`
    restricts to a subset of each group's seeds (smoke runs; default: all)."""
    indexed = [
        (phase_index(g, manifest.primary_nq), i, g) for i, g in enumerate(manifest.groups) if g.part in parts
    ]
    indexed.sort(key=lambda t: (t[0], t[1]))
    return [
        Job(g, seed) for _, _, g in indexed for seed in g.seeds if seeds is None or seed in seeds
    ]


def count_by_part(jobs: Sequence[Job]) -> dict[str, int]:
    counts = dict.fromkeys(PARTS, 0)
    for job in jobs:
        counts[job.group.part] += 1
    return counts


# --- paths / skip-existing ----------------------------------------------------


def leaf_glob(group: RunGroup, seed: int) -> str:
    """Glob (relative to the group's `dir_results`) matching this job's leaf dir."""
    return (
        f"*/SEED_{seed}/NQ_{group.n_query}_NIL_{group.n_init_labeled}_NR_{group.n_round}_NE_*/"
        f"{group.strategy_name}"
    )


def find_leaf(group: RunGroup, seed: int, *, base: Path | None = None) -> Path | None:
    """The existing leaf directory of a job (any dataset folder / n_epoch), or None.

    `base` overrides the directory `dir_results` is resolved against (default
    the repo root); used by the report/tests to point at another tree.
    """
    root = (base if base is not None else REPO_ROOT) / group.dir_results
    if not root.is_dir():
        return None
    matches = sorted(root.glob(leaf_glob(group, seed)))
    return matches[0] if matches else None


def job_done(job: Job) -> bool:
    leaf = find_leaf(job.group, job.seed)
    return leaf is not None and (leaf / "results.json").is_file()


# --- running ------------------------------------------------------------------


def build_command(job: Job, manifest: Manifest, device: str, python: str | None = None) -> list[str]:
    g = job.group
    return [
        python or sys.executable,
        "trainer.py",
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
    """Same env handling as scripts/ablations/run_ablation_gpu_*.sh."""
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
    """Execute jobs sequentially; a failing job is logged and the batch continues."""
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
        if skip_existing and job_done(job):
            summary.skipped += 1
            log(f"{head} SKIP (results.json exists)")
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


def verify_job(job: Job, *, base: Path | None = None) -> JobStatus:
    leaf = find_leaf(job.group, job.seed, base=base)
    if leaf is None:
        return JobStatus(job, "MISSING")
    missing, warnings = check_leaf(leaf)
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
    status: str  # PASS | FAIL | NO_DATA
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
            audits.append(SeedAudit(seed, "NO_DATA", len(runs), unreadable=unreadable))
            continue
        reference_id, reference = readable[0]
        offenders = tuple(rid for rid, idxs in readable if idxs != reference)
        audits.append(SeedAudit(
            seed, "FAIL" if offenders else "PASS", len(readable), reference_id, offenders, unreadable,
        ))
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


def render_job_table(jobs: Sequence[Job], primary_nq: int) -> str:
    lines = [f"{'#':>4}  {'phase':<16} {'run id':<44} {'strategy':<24} {'nq':>4} {'nil':>5} {'nr':>3} seed"]
    for n, job in enumerate(jobs, start=1):
        g = job.group
        lines.append(
            f"{n:>4}  {PHASE_LABELS[phase_index(g, primary_nq)]:<16} {g.id:<44} {g.strategy_name:<24} "
            f"{g.n_query:>4} {g.n_init_labeled:>5} {g.n_round:>3} {job.seed}"
        )
    counts = count_by_part(jobs)
    lines.append("")
    for part in PARTS:
        groups = len({j.group.id for j in jobs if j.group.part == part})
        lines.append(f"  {part:<12} {counts[part]:>4} runs ({groups} run groups)")
    lines.append(f"  {'TOTAL':<12} {len(jobs):>4} runs ({len({j.group.id for j in jobs})} run groups)")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dalmax.campaign", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", default=None, help=f"manifest path (default {DEFAULT_MANIFEST})")
        p.add_argument("--micro", action="store_true", help=f"use {DEFAULT_MICRO_MANIFEST} (CPU smoke)")
        p.add_argument("--part", default="all", help="paper1,upper_bound,rnhal,texhal (comma-separated) or all")
        p.add_argument("--seeds", default=None, help="comma-separated subset of the seeds (smoke); default all")

    common(sub.add_parser("list", help="print the job table and counts"))
    run_p = sub.add_parser("run", help="execute jobs sequentially via trainer.py")
    common(run_p)
    run_p.add_argument("--dry-run", action="store_true", help="echo commands, execute nothing")
    run_p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="skip jobs whose leaf results.json exists (default on)",
    )
    run_p.add_argument("--device", default=None, choices=["cuda", "cpu", "auto"],
                       help="default: cuda (cpu with --micro)")
    common(sub.add_parser("verify", help="OK/INCOMPLETE/MISSING per job"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest_path = args.manifest or (DEFAULT_MICRO_MANIFEST if args.micro else DEFAULT_MANIFEST)
        manifest = load_manifest(manifest_path)
        parts = parse_parts(args.part)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    jobs = expand_jobs(manifest, parts, seeds)
    if args.command == "list":
        print(f"manifest: {manifest.path}  results_root: {manifest.results_root}")
        print(render_job_table(jobs, manifest.primary_nq))
        return 0
    if args.command == "verify":
        statuses = [verify_job(j) for j in jobs]
        print(render_verify(statuses))
        audits = audit_seed_consistency(jobs)
        print()
        print(render_seed_audit(audits))
        artifacts_ok = all(s.status == "OK" for s in statuses)
        seeds_ok = all(a.status == "PASS" for a in audits)
        return 0 if artifacts_ok and seeds_ok else 1

    device = args.device or ("cpu" if args.micro else "cuda")
    summary = run_jobs(
        jobs, manifest, device=device, skip_existing=args.skip_existing, dry_run=args.dry_run
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
