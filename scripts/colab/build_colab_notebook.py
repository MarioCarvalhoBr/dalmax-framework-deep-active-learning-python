"""Generate `notebooks/colab_runbook.ipynb`, the runnable companion of `COLAB_RUNBOOK.md`.

Until 2026-11-11 the notebook was edited by hand; since ADR 0016 it is generated from the cell list
below so that its gates and expected counts cannot drift from the manifests (the expected runs /
groups per GPU part are computed from `files_config/campaign/manifest.json` and
`manifest_kmh_sweep.json` at generation time). `tests/test_colab_notebook.py` fails if the committed
notebook differs from what this script emits. Run from the repo root:

    poetry run python scripts/colab/build_colab_notebook.py            # rewrite
    poetry run python scripts/colab/build_colab_notebook.py --check    # exit 1 on drift

The notebook runs ONE GPU part of the clean campaign on `DATA/daninhas_full_fixed` per Colab account
(`GPU_PART`, ADR 0016). Its gate cells raise (so "Run all" stops) on a wrong `GPU_PART`, an unfilled
`DRIVE_ROOT`, a missing or wrong dataset, a dirty checkout, results of the other GPU part on this
Drive, a job count that differs from the manifests, too little free Drive space for the remaining
runs (`make campaign-space-gpu-part`), or an unreviewed pilot (`PILOT_REVIEWED`, 2026-11-12: the pilot
runs seed 1 of the part's VAAL/WAAL groups and one cheap strategy before the long runs).

Every session checks out ONE pinned ref (`GIT_REF`, default `DEFAULT_GIT_REF`, a tag created after the
merge) detached, instead of pulling a branch, so a push during the campaign cannot mix code versions;
the run cell prints a non-zero exit code of `make campaign-run-gpu-part` and points to the logs.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dalmax.campaign import (  # noqa: E402
    GPU_PARTS,
    load_estimates,
    load_manifest,
    parse_selection,
    pilot_group_ids,
    plan_jobs,
    selected_groups,
)

OUT = "notebooks/colab_runbook.ipynb"
MANIFESTS = ("files_config/campaign/manifest.json", "files_config/campaign/manifest_kmh_sweep.json")
REPO_URL = "https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python.git"
# The tag both accounts check out for the whole campaign. v2 (2026-11-12) = v1 + the WAAL control
# `WAALRandomQuery` and the `daninhas_full_fixed` metadata layout (logs/); v1 predates both.
DEFAULT_GIT_REF = "campaign-fixed-v2"
RUN_EXIT_FILE = "/content/campaign_run_exit_code"  # local runtime disk, not the repo or Drive
DRIVE_ROOT_PLACEHOLDER = "/content/drive/MyDrive/<FOLDER_THAT_HOLDS_DATA>"
DATASET = "daninhas_full_fixed"
# Images per split of daninhas_full_fixed (scripts/colab/setup_colab.sh checks the same counts); the folder also
# holds README.txt, MANIFEST.csv and logs/ (2 files), which `make dataset-check` verifies against the images.
DATASET_SPLIT_IMAGES: dict[str, int] = {"train": 7230, "test": 1008}


def expected_counts() -> dict[str, tuple[int, int]]:
    """`{gpu part: (runs, run groups)}` over both full-scale manifests."""
    manifests = [load_manifest(REPO_ROOT / path, validate=False) for path in MANIFESTS]
    counts: dict[str, tuple[int, int]] = {}
    for gpu_part in GPU_PARTS:
        jobs = [j for seg in plan_jobs(manifests, parse_selection(gpu_part)) for j in seg.jobs]
        counts[gpu_part] = (len(jobs), len({j.group.id for j in jobs}))
    return counts


def manifest_totals() -> list[tuple[int, int]]:
    """`(runs, run groups)` of each manifest of `MANIFESTS`, every group and seed."""
    totals = []
    for path in MANIFESTS:
        groups = load_manifest(REPO_ROOT / path, validate=False).groups
        totals.append((sum(len(g.seeds) for g in groups), len(groups)))
    return totals


def results_roots() -> list[str]:
    return [load_manifest(REPO_ROOT / path, validate=False).results_root for path in MANIFESTS]


def pilot_hours() -> dict[str, float]:
    """`{gpu part: estimated GPU hours of its pilot}` (seed 1 of `pilot_group_ids`)."""
    manifests = [load_manifest(REPO_ROOT / path, validate=False) for path in MANIFESTS]
    estimates: dict[str, float] = {}
    for manifest in manifests:
        estimates.update(load_estimates(manifest))
    hours: dict[str, float] = {}
    for gpu_part in GPU_PARTS:
        ids = pilot_group_ids(selected_groups(manifests, parse_selection(gpu_part)))
        hours[gpu_part] = sum(estimates[gid] for gid in ids) / 3600
    return hours


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip("\n")}


def code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(text).strip("\n")}


def cells() -> list[dict[str, Any]]:
    expected = expected_counts()
    roots = results_roots()
    (main_runs, main_groups), (sweep_runs, sweep_groups) = manifest_totals()
    pilot = pilot_hours()
    part1, part2 = GPU_PARTS
    expected_literal = "{" + ", ".join(f'"{p}": {expected[p]}' for p in GPU_PARTS) + "}"
    return [
        md(f"""
            # DalMax -- Google Colab Pro runbook (notebook form)

            Runnable companion to [`COLAB_RUNBOOK.md`](https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python/blob/main/COLAB_RUNBOOK.md),
            section 6 ("The clean campaign on `{DATASET}`", ADR 0016). **`COLAB_RUNBOOK.md` is the
            source of truth.** This notebook is generated by `scripts/colab/build_colab_notebook.py`; do not edit
            it by hand.

            One Colab account runs ONE GPU part of the campaign (both manifests: the main campaign and the
            KMH sweep): account A `GPU_PART = "{part1}"`, account B `GPU_PART = "{part2}"`, in parallel.

            **Never run this notebook on the local no-GPU dev notebook** (`.claude/rules/data-safety.md`).

            Fill in the **Parameters** cell right below (the only cell you edit), then run the cells top to
            bottom. The gate cells raise on any problem, so "Run all" stops there instead of spending credits.
            """),
        code(f'''
            import os

            # Parameters -- the ONLY cell in this notebook you edit.
            DRIVE_ROOT = "{DRIVE_ROOT_PLACEHOLDER}"  # REQUIRED, per account: the Drive folder holding DATA/{DATASET}/
            REPO_URL = "{REPO_URL}"
            REPO_DIR = "/content/dalmax"
            # The code version of the WHOLE campaign: a tag (or a commit SHA), checked out detached on every session
            # and on both accounts, so a later push cannot mix code versions. Do not use a branch name here.
            GIT_REF = "{DEFAULT_GIT_REF}"
            GPU_PART = "{part1}"  # account A: "{part1}"; account B: "{part2}" (ADR 0016). Re-running resumes.
            # Set to True only after reading the pilot report of section 6 (wall times vs estimates, no degeneracy
            # flag); until then "Run all" stops right before the full run.
            PILOT_REVIEWED = False

            # Expected size of each GPU part over BOTH manifests (computed from the manifests when the notebook
            # was generated); the gate cell before the run asserts it.
            EXPECTED = {expected_literal}  # (runs, run groups)
            RESULTS_ROOTS = {roots!r}
            DATASET = "{DATASET}"
            DATASET_SPLIT_IMAGES = {DATASET_SPLIT_IMAGES!r}  # images per split

            # --- gate: fail fast on a wrong value ---
            if GPU_PART not in EXPECTED:
                raise ValueError(f"GPU_PART must be one of {{sorted(EXPECTED)}}, got {{GPU_PART!r}}")
            if "<" in DRIVE_ROOT or not DRIVE_ROOT.startswith("/content/drive/"):
                raise ValueError("Fill in DRIVE_ROOT: the folder of THIS account's Drive that holds DATA/" + DATASET)
            OTHER_GPU_PART = next(p for p in EXPECTED if p != GPU_PART)
            EXPECTED_RUNS, EXPECTED_GROUPS = EXPECTED[GPU_PART]

            # Export for every `!`-shell cell and for scripts/colab/setup_colab.sh (DRIVE_ROOT is required there).
            os.environ["DRIVE_ROOT"] = DRIVE_ROOT
            os.environ["GPU_PART"] = GPU_PART

            print("DRIVE_ROOT:", DRIVE_ROOT)
            print("REPO_URL:  ", REPO_URL)
            print("GIT_REF:   ", GIT_REF)
            print()
            print("#" * 78)
            print(f"#  GPU PART: {{GPU_PART}}  ->  expect {{EXPECTED_RUNS}} runs ({{EXPECTED_GROUPS}} run groups)")
            print("#  Account A runs {part1}; account B runs {part2}. Wrong account? Edit GPU_PART, re-run.")
            print("#" * 78)
            '''),
        md("""
            ## 0. Runtime check

            See `COLAB_RUNBOOK.md` section 0. Select a GPU runtime (any); the cell below lists whichever GPU
            Colab assigned. Its name is recorded automatically in every run's `run_metadata.json` (`environment.gpus`).
            """),
        code("!nvidia-smi"),
        code('''
            import sys

            # Informational only: Colab's SYSTEM Python may be 3.13; the project runs in a Poetry venv on a
            # uv-managed Python 3.12 (section 3), validated right after the install.
            print(f"Kernel (system) Python: {sys.version.split()[0]} -- any version is OK here")
            '''),
        md("""
            ## 1. Mount Google Drive

            See `COLAB_RUNBOOK.md` section 1. The dataset must be on THIS account's Drive.
            """),
        code('''
            from google.colab import drive

            drive.mount("/content/drive")
            '''),
        code('''
            # Gate: the canonical dataset (folder or its zip) is on this Drive; the old daninhas_full is not used.
            folder = f"{DRIVE_ROOT}/DATA/{DATASET}"
            zip_path = f"{DRIVE_ROOT}/DATA/{DATASET}.zip"
            if not (os.path.isdir(folder) or os.path.isfile(zip_path)):
                raise FileNotFoundError(
                    f"Neither {folder} nor {zip_path} exists. Upload DATA/{DATASET}/ to this account's Drive under "
                    "DRIVE_ROOT (COLAB_RUNBOOK.md section 6, step 1) or fix DRIVE_ROOT in the Parameters cell."
                )
            print("Dataset found on Drive:", folder if os.path.isdir(folder) else zip_path)
            '''),
        md("""
            ## 2. Clone the repo and check out `GIT_REF`

            Clones into `/content/dalmax` (local runtime disk, not Drive) if this runtime does not have it yet, then on
            EVERY session fetches the tags and checks out `GIT_REF` detached (no branch is pulled, so both accounts and
            every session run the same commit, printed below). The gate refuses a checkout with modified tracked files.
            See `COLAB_RUNBOOK.md` section 2.
            """),
        code('''
            import subprocess

            if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
                print(f"Cloning {REPO_URL} into {REPO_DIR}...")
                subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

            # Gate: no modified tracked file (results/ symlink and DATA/ are untracked and do not count).
            dirty = subprocess.run(["git", "-C", REPO_DIR, "status", "--porcelain", "--untracked-files=no"],
                                   check=True, capture_output=True, text=True).stdout.strip()
            if dirty:
                raise RuntimeError("The checkout has local modifications; reset them before running:\\n" + dirty)

            # Pin the code version: the same GIT_REF on every session (no `git pull` of a branch).
            subprocess.run(["git", "-C", REPO_DIR, "fetch", "--tags", "origin"], check=True)
            subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", GIT_REF], check=True)
            head = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            wanted = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", f"{GIT_REF}^{{commit}}"],
                                    check=True, capture_output=True, text=True).stdout.strip()
            if head != wanted:
                raise RuntimeError(f"HEAD is {head}, but {GIT_REF} resolves to {wanted}")
            print(f"GIT_REF {GIT_REF} -> commit {head} (every run records it as git_commit in run_metadata.json)")

            os.chdir(REPO_DIR)
            %cd $REPO_DIR
            !git log --oneline -1
            '''),
        md("""
            ## 3. Install dependencies with Poetry

            See `COLAB_RUNBOOK.md` section 3.
            """),
        code('''
            # Colab's system Python may be 3.13, which torch==2.5.0 has no wheels for.
            # Install a separate Python 3.12 with uv and point Poetry at it (idempotent).
            !pip install -q uv poetry
            !uv python install 3.12
            '''),
        code('''
            %cd {REPO_DIR}
            !poetry env use "$(uv python find 3.12)"
            !poetry install
            '''),
        code('''
            import subprocess

            # Gate: the Poetry venv interpreter (not the kernel's own Python) is 3.12 and sees a GPU.
            probe = subprocess.run(
                ["poetry", "run", "python", "-c",
                 "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"],
                capture_output=True, text=True, cwd=REPO_DIR,
            )
            print(probe.stdout.strip(), probe.stderr.strip()[-500:])
            fields = probe.stdout.split()
            venv_version = fields[0] if probe.returncode == 0 and fields else ""
            if not venv_version.startswith("3.12"):
                raise RuntimeError(
                    f"Poetry venv Python is {venv_version or 'unavailable'}, expected 3.12.x: re-run the previous "
                    'cells (`uv python install 3.12`, `poetry env use "$(uv python find 3.12)"`, `poetry install`).'
                )
            if fields[-1:] != ["True"]:
                raise RuntimeError("torch sees no GPU: Runtime -> Change runtime type -> GPU, then start again.")
            print(f"Poetry venv Python {venv_version} OK, CUDA available")
            '''),
        md(f"""
            ## 4. Dataset and results/ (Drive)

            `make colab-setup` runs `scripts/colab/setup_colab.sh`: it builds `{DATASET}.zip` on Drive on the
            first session only, unzips it onto the local runtime disk, checks the image count per split
            ({DATASET_SPLIT_IMAGES["train"]} train, {DATASET_SPLIT_IMAGES["test"]} test), that the files on disk are
            exactly the rows of its `MANIFEST.csv` (name and MD5) and that no MD5 repeats
            (`scripts/datasets/build_daninhas_full_fixed.py --check`), and symlinks `results/` to
            `$DRIVE_ROOT/results`. Idempotent. See `COLAB_RUNBOOK.md` section 4.
            """),
        code('''
            import subprocess

            subprocess.run(["make", "colab-setup"], cwd=REPO_DIR, check=True)  # raises on any setup failure
            '''),
        code('''
            import subprocess

            # Gate: the local dataset copy is the canonical one (images per split, then disk == MANIFEST.csv,
            # the metadata layout and no repeated MD5).
            for split, expected_images in DATASET_SPLIT_IMAGES.items():
                n_images = sum(len(files) for _, _, files in os.walk(f"{REPO_DIR}/DATA/{DATASET}/{split}"))
                if n_images != expected_images:
                    raise RuntimeError(f"DATA/{DATASET}/{split} has {n_images} files, expected {expected_images}")
            subprocess.run(["make", "dataset-check"], cwd=REPO_DIR, check=True)
            print(f"DATA/{DATASET}: {DATASET_SPLIT_IMAGES} images, matches MANIFEST.csv, no repeated MD5")
            '''),
        md("""
            ## 5. Sanity checks

            See `COLAB_RUNBOOK.md` section 5: the fast tests, then one short real-data GPU run into
            `results/colab_check/` (never part of the campaign trees).
            """),
        code('''
            import subprocess

            subprocess.run(["make", "test"], cwd=REPO_DIR, check=True)
            '''),
        code('''
            import subprocess

            subprocess.run(["make", "colab-check"], cwd=REPO_DIR, check=True)
            '''),
        md("""
            ### GPU identity check

            Confirm this is the GPU you intended **before launching the campaign**. The cell below reads the
            `run_metadata.json` written by the check run above (`environment` and `determinism` blocks).
            """),
        code('''
            import glob
            import json

            hits = sorted(glob.glob(f"results/colab_check/{DATASET}/**/run_metadata.json", recursive=True))
            assert hits, f"no run_metadata.json under results/colab_check/{DATASET}/ -- did `make colab-check` finish?"
            meta = json.load(open(hits[-1]))
            env = meta["environment"]
            gpus = env.get("gpus") or []
            assert gpus, "no GPU recorded in run_metadata.json -- is this a GPU runtime?"
            assert meta["config"]["dataset"]["data_dir"] == f"DATA/{DATASET}/", meta["config"]["dataset"]["data_dir"]
            gpu = gpus[0]
            print("GPU name:      ", gpu.get("name"))
            print("GPU memory GB: ", gpu.get("total_memory_gb"))
            print("driver:        ", gpu.get("driver_version"))
            print("CUDA (torch):  ", env["torch"].get("cuda"), "| cuDNN:", env["torch"].get("cudnn"))
            print("determinism:   ", meta["determinism"])
            '''),
        code(f"!poetry run python tools/loader.py --model results/colab_check/{DATASET}/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth"),
        md(f"""
            ## 6. The clean campaign on `{DATASET}` (ADR 0016)

            Everything the three papers need, re-run on the duplicate-free dataset: {main_runs + sweep_runs} runs /
            {main_groups + sweep_groups} run groups over two manifests (main campaign {main_runs} / {main_groups} into
            `{roots[0]}/`, including the 3 runs of the WAAL control `WAALRandomQuery`; KMH sweep {sweep_runs} /
            {sweep_groups} into `{roots[1]}/`), split into two GPU parts of about equal estimated GPU time, one per
            Colab account:

            | GPU part | Account | Runs | Run groups |
            |---|---|---|---|
            | `{part1}` | A | {expected[part1][0]} | {expected[part1][1]} |
            | `{part2}` | B | {expected[part2][0]} | {expected[part2][1]} |

            Inside a part the order is: cheap paper-1 groups, the ablations, the KMH sweep, then the long baselines
            (`AdversarialBIM`, `VAAL`/`WAAL` and the WAAL control, `AdversarialDeepFool`) last. Re-running the run cell resumes (complete
            leaves are skipped). Estimated hours per part: printed by the list cell. Before the full run: the pre-run
            gate (job count, no leaf of the other part, free Drive space for the remaining runs: about 96 MB per run
            plus 20%), then the **pilot** (seed 1 of this part's `VAAL` and `WAAL` groups at its largest budget and one
            cheap strategy) and its report. See `COLAB_RUNBOOK.md` section 6.
            """),
        code("!make campaign-list-gpu-part GPU_PART={GPU_PART} 2>&1 | tail -n 40"),
        code('''
            import re
            import subprocess

            # Pre-run gate: every failure RAISES, so "Run all" stops here instead of launching a wrong part.
            listing = subprocess.run(["make", "campaign-list-gpu-part", f"GPU_PART={GPU_PART}"], cwd=REPO_DIR,
                                     check=True, capture_output=True, text=True).stdout
            combined = listing.split("Combined:")[-1]
            total = re.search(r"^\\s*TOTAL\\s+(\\d+) runs \\((\\d+) run groups\\)", combined, flags=re.M)
            assert total, "campaign-list printed no TOTAL line:\\n" + listing[-2000:]
            assert (int(total.group(1)), int(total.group(2))) == (EXPECTED_RUNS, EXPECTED_GROUPS), (
                f"{GPU_PART}: campaign-list reports {total.group(1)} runs ({total.group(2)} run groups), "
                f"expected {EXPECTED_RUNS} ({EXPECTED_GROUPS})"
            )
            # Dirty state: this account's results must not hold leaves of the OTHER GPU part.
            other = subprocess.run(["make", "campaign-verify-gpu-part", f"GPU_PART={OTHER_GPU_PART}"], cwd=REPO_DIR,
                                   capture_output=True, text=True).stdout
            found = re.search(r"Totals: OK=(\\d+) INCOMPLETE=(\\d+)", other)
            assert found, "campaign-verify printed no Totals line:\\n" + other[-2000:]
            if int(found.group(1)) + int(found.group(2)):
                raise RuntimeError(f"results of {OTHER_GPU_PART} are on this Drive ({found.group(0)}): wrong account or "
                                   "wrong GPU_PART? Stop and check before spending credits.")
            # Drive quota: free space for every remaining run of the part (96 MB per run + 20%), else the batch would
            # train for hours and fail at every save. `campaign run` repeats this check and stops on a full disk.
            space = subprocess.run(["make", "campaign-space-gpu-part", f"GPU_PART={GPU_PART}"], cwd=REPO_DIR,
                                   capture_output=True, text=True)
            print(space.stdout.strip())
            if space.returncode != 0:
                raise RuntimeError("Not enough free Drive space for the remaining runs of this part (numbers above): free "
                                   "space on this account's Drive first.\\n" + space.stderr[-1000:])
            print(total.group(0).strip(), f"-- gate OK for {GPU_PART}. Next: the pilot.")
            '''),
        md(f"""
            **Pilot** (seed 1 of this part's `VAAL` and `WAAL` groups at its largest budget, plus `RandomSampling` at
            that budget; estimated {pilot[part1]:.1f} GPU h in `{part1}`, {pilot[part2]:.1f} GPU h in `{part2}`).
            The VAAL/WAAL costs are FLOP estimates, so measure them before the long runs. The pilot leaves are normal
            campaign leaves: the full run skips them. Then read the report: wall time against the estimate (and the
            projected hours of each strategy over the part), the weighted F1 per round, and every
            `strategy_diagnostics` series with degeneracy flags (`pool_order_fraction` near 1 = the KI-45 pool-order
            signature; constant scores; zero spread). If the times or the diagnostics look wrong, stop and report
            before spending the part's credits. Otherwise set `PILOT_REVIEWED = True` in the Parameters cell, re-run
            it and continue from the gate cell below.
            """),
        code("!make campaign-pilot GPU_PART={GPU_PART}"),
        code("!make campaign-pilot-report GPU_PART={GPU_PART}"),
        md("Run (skip-existing: **re-running this cell resumes after a disconnect** -- redo sections 0-5 first)."),
        code('''
            # Gate: the full run starts only after the pilot report was read.
            if not PILOT_REVIEWED:
                raise RuntimeError("Read the pilot report above, then set PILOT_REVIEWED = True in the Parameters cell, "
                                   "re-run that cell and continue from here.")
            print(f"Pilot reviewed -- launching the full {GPU_PART} run (completed pilot leaves are skipped).")
            '''),
        code(f'''
            import time

            start = time.time()
            !rm -f {RUN_EXIT_FILE}
            !make campaign-run-gpu-part GPU_PART={{GPU_PART}}; echo $? > {RUN_EXIT_FILE}
            '''),
        code(f'''
            import subprocess

            elapsed_hours = (time.time() - start) / 3600
            print(f"{{GPU_PART}}: this session ran {{elapsed_hours:.2f}} hours.")
            # Surface a failed run. make exits 2 whenever `python -m dalmax.campaign run` fails; the campaign's own code
            # is on the `make: *** [...] Error N` line above: 4 = the Drive is short of space (nothing was started, or
            # the batch stopped), 1 = at least one run failed (the batch went on with the others).
            run_exit = open("{RUN_EXIT_FILE}").read().strip() if os.path.exists("{RUN_EXIT_FILE}") else "unknown"
            if run_exit != "0":
                space = subprocess.run(["make", "campaign-space-gpu-part", f"GPU_PART={{GPU_PART}}"], cwd=REPO_DIR,
                                       capture_output=True, text=True)
                print("#" * 78)
                print(f"#  make campaign-run-gpu-part exited with code {{run_exit}} (the campaign's code: `Error N` above)")
                if space.returncode != 0:
                    print("#  The Drive is SHORT OF SPACE for the remaining runs (exit 4): free space on this account's")
                    print("#  Drive, then re-run the run cell (it resumes).")
                    print(space.stdout.strip())
                else:
                    print("#  Some runs failed: read failures.log / campaign.log, then re-run the run cell (it resumes).")
                for root in RESULTS_ROOTS:
                    print(f"#  logs: {{REPO_DIR}}/{{root}}/campaign.log and {{REPO_DIR}}/{{root}}/failures.log")
                print("#" * 78)
                raise RuntimeError(f"make campaign-run-gpu-part exited with code {{run_exit}} (see the lines above)")
            print("make campaign-run-gpu-part exited with code 0")
            '''),
        md("**Progress** (usable anytime, e.g. after a disconnect):"),
        code('''
            !make campaign-verify-gpu-part GPU_PART={GPU_PART} 2>&1 | grep -E "^Totals|seed [0-9]|distinct order|distinct commit|WARN:"
            for root in RESULTS_ROOTS:
                log = f"{REPO_DIR}/{root}/campaign.log"
                tail = open(log).read().splitlines()[-5:] if os.path.exists(log) else ["no campaign.log yet"]
                print(f"--- {root}/campaign.log (last lines)", *tail, sep="\\n")
            '''),
        code('''
            for root in RESULTS_ROOTS:
                log = f"{REPO_DIR}/{root}/failures.log"
                print(open(log).read() if os.path.exists(log) else f"{root}: no failures logged")
            '''),
        md("""
            **Verify**: OK/INCOMPLETE/MISSING per job, the seed-consistency audit (every run of a seed must start
            from the identical initial labeled IMAGES, compared by path) and the data-order audit (every leaf must
            index the same ordered image lists, KI-47). A leaf with an integrity problem (truncated or empty file, wrong
            round count) is `INCOMPLETE (integrity: ...)` and is re-run by the run cell. The code-version audit lists the
            `git_commit` of the OK leaves: it must show ONE commit (the one printed by section 2) and no `WARN:` line.
            Exit code 0 = all OK, 1 = an incomplete/missing job or a FAIL audit, 2 = only audit WARNs. Re-run the run
            cell to redo failed or missing jobs.
            """),
        code("!make campaign-verify-gpu-part GPU_PART={GPU_PART}"),
        md("""
            **Zip this GPU part** for download (both results roots, this account's `campaign.log` / `failures.log`
            renamed to `campaign_<GPU_PART>.log` / `failures_<GPU_PART>.log` so the two accounts never collide). The
            cell refuses (raises) unless verify reports every run of the part `OK` (`ZIP_PARTIAL = True` ships an
            incomplete part on purpose) and checks the free space first: the checkpoints do not compress, so the
            zip is about as large as the results (~95 MB per run).
            """),
        code('''
            import re
            import shutil
            import subprocess

            ZIP_PARTIAL = False  # True only to ship an incomplete part on purpose (e.g. no credits left)
            verify = subprocess.run(["make", "campaign-verify-gpu-part", f"GPU_PART={GPU_PART}"], cwd=REPO_DIR,
                                    capture_output=True, text=True)
            totals = re.search(r"Totals: OK=(\\d+) .*", verify.stdout)
            print(totals.group(0) if totals else verify.stdout[-2000:] + verify.stderr[-2000:])
            n_ok = int(totals.group(1)) if totals else 0
            if n_ok != EXPECTED_RUNS and not ZIP_PARTIAL:
                raise RuntimeError(f"verify reports OK={n_ok}, expected {EXPECTED_RUNS}: re-run the run cell (resumes) first")
            zip_path = f"/content/{GPU_PART}.zip"
            need = sum(os.path.getsize(os.path.join(d, f)) for root in RESULTS_ROOTS
                       for d, _, files in os.walk(f"{REPO_DIR}/{root}") for f in files)
            free = min(shutil.disk_usage("/content").free, shutil.disk_usage(DRIVE_ROOT).free)
            print(f"results to zip: {need / 1e9:.1f} GB; free (local disk / Drive, the smaller): {free / 1e9:.1f} GB")
            if free < 1.1 * need:
                raise RuntimeError("Not enough free space for the zip: free Drive space, or download the two results "
                                   "folders from Drive directly (COLAB_RUNBOOK.md section 6, step 9).")
            excluded = " ".join(f"-x '{root}/campaign.log' -x '{root}/failures.log'" for root in RESULTS_ROOTS)
            subprocess.run(f"rm -f {zip_path} && zip -qr {zip_path} {' '.join(RESULTS_ROOTS)} {excluded}",
                           shell=True, cwd=REPO_DIR, check=True)
            stage = f"/content/zip_stage_{GPU_PART}"
            shutil.rmtree(stage, ignore_errors=True)
            for root in RESULTS_ROOTS:
                os.makedirs(f"{stage}/{root}")
                for name in ("campaign", "failures"):
                    log = f"{REPO_DIR}/{root}/{name}.log"
                    if os.path.exists(log):
                        shutil.copy(log, f"{stage}/{root}/{name}_{GPU_PART}.log")
            subprocess.run(f"cd {stage} && zip -qr {zip_path} results", shell=True, check=True)
            listing = subprocess.run(["unzip", "-l", zip_path], capture_output=True, text=True, check=True).stdout
            n_models = listing.count("saved_model.pth")
            print(f"{zip_path}: {n_models} checkpoints")
            if n_models != (n_ok if ZIP_PARTIAL else EXPECTED_RUNS):
                raise RuntimeError(f"zip holds {n_models} checkpoints, expected {n_ok if ZIP_PARTIAL else EXPECTED_RUNS}")
            subprocess.run(["cp", zip_path, f"{DRIVE_ROOT}/"], check=True)
            subprocess.run(["ls", "-lh", f"{DRIVE_ROOT}/{GPU_PART}.zip"], check=True)
            '''),
        md("""
            **Optional: partial report** of this account's part (the tables of the full campaign are built locally
            after merging both parts, `COLAB_RUNBOOK.md` section 6, step 10). Rows of the other part show `TBD`.
            """),
        code("!make campaign-report-all"),
        md("""
            ## 7. Troubleshooting

            See `COLAB_RUNBOOK.md` section 7. Quick pointers:

            - Session dropped / limit reached -> new runtime, Parameters cell unchanged, sections 0-5, then the run
              cell again (complete leaves are skipped; at most the run in progress is lost).
            - `STOP no space` in `campaign.log` / exit 4 -> the Drive is full: free space on this account's Drive, then
              re-run the run cell (it resumes; the failed run is retried).
            - Drive FUSE `Transport endpoint is not connected` -> re-run the mount cell and `make colab-setup`.
            - `results/` symlink conflicts -> read `scripts/colab/setup_colab.sh`'s error message (it names the path).
            - Out of GPU memory -> stop and report it; never edit a committed params file to lower the batch size.
            """),
    ]


def notebook() -> dict[str, Any]:
    cell_list = cells()
    for index, cell in enumerate(cell_list):
        cell["id"] = f"cell-{index:02d}"
        lines = cell["source"].split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cells": cell_list,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "colab_runbook.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed notebook differs")
    args = parser.parse_args()
    path = REPO_ROOT / OUT
    text = render()
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"DRIFT: {OUT}")
            return 1
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
