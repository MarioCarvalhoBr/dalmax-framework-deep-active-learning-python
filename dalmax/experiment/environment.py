"""Best-effort capture of the execution environment (OS, CPU/RAM, GPU, CUDA).

Everything is probed at runtime; every probe is wrapped so a failure yields
`None` instead of crashing a run. The hostname is deliberately NOT recorded
(`.claude/rules/data-safety.md`). Importing this module has no side effects.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
from collections.abc import Callable
from typing import Any

import torch

_NVIDIA_SMI_QUERY = "index,name,driver_version,memory.total"


def _safe(fn: Callable[[], Any]) -> Any:
    """Run `fn`; return `None` on any exception."""
    try:
        return fn()
    except Exception:
        return None


def _os_pretty_name() -> str | None:
    with open("/etc/os-release", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_total_gb() -> float | None:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 2)
    return None


def _nvidia_smi() -> dict[int, dict[str, Any]]:
    """Parse `nvidia-smi` CSV output into `{index: {...}}`; `{}` if unavailable."""
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        rows[int(parts[0])] = {
            "nvidia_smi_name": parts[1],
            "driver_version": parts[2],
            "memory_total_mib": int(float(parts[3])),
        }
    return rows


def _is_colab() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):  # parent package `google` absent
        return False


def _gpus() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    smi = _safe(_nvidia_smi) or {}
    gpus: list[dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        props = _safe(lambda i=idx: torch.cuda.get_device_properties(i))
        total = _safe(lambda p=props: round(p.total_memory / 1024**3, 2))
        capability = _safe(lambda p=props: f"{p.major}.{p.minor}")
        entry = {
            "index": idx,
            "name": _safe(lambda i=idx: torch.cuda.get_device_name(i)),
            "total_memory_gb": total,
            "compute_capability": capability,
            "multi_processor_count": _safe(lambda p=props: int(p.multi_processor_count)),
            "driver_version": None,
            "nvidia_smi_name": None,
            "memory_total_mib": None,
        }
        entry.update(smi.get(idx, {}))
        gpus.append(entry)
    return gpus


def collect_environment() -> dict[str, Any]:
    """Return a JSON-serializable description of the current execution environment."""
    return {
        "os": {
            "system": _safe(platform.system),
            "release": _safe(platform.release),
            "version": _safe(platform.version),
            "platform": _safe(platform.platform),
            "distro": _safe(_os_pretty_name),
        },
        "machine": {
            "arch": _safe(platform.machine),
            "cpu_model": _safe(_cpu_model),
            "cpu_count": _safe(os.cpu_count),
            "ram_total_gb": _safe(_ram_total_gb),
        },
        "python": {
            "version": _safe(platform.python_version),
            "implementation": _safe(platform.python_implementation),
        },
        "torch": {
            "version": _safe(lambda: torch.__version__),
            "cuda": _safe(lambda: torch.version.cuda),
            "cudnn": _safe(torch.backends.cudnn.version),
            "cuda_available": _safe(torch.cuda.is_available),
        },
        "gpus": _safe(_gpus) or [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": _safe(
            lambda: torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "runtime": {
            "is_colab": _safe(_is_colab),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        },
    }


def summarize_environment(env: dict[str, Any]) -> str:
    """One-line human summary for the run log."""
    os_name = (env.get("os") or {}).get("distro") or (env.get("os") or {}).get("platform")
    gpus = env.get("gpus") or []
    if gpus:
        g = gpus[0]
        gpu = (
            f"GPU0: {g.get('name')} {g.get('total_memory_gb')} GB driver {g.get('driver_version')}"
        )
    else:
        gpu = "GPU: none"
    torch_info = env.get("torch") or {}
    return f"Environment: {os_name} | {gpu} | torch {torch_info.get('version')} (CUDA {torch_info.get('cuda')})"
