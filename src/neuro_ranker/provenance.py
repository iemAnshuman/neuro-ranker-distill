from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "onnx",
    "onnxruntime",
    "fastapi",
    "uvicorn",
)


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _sha256_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    embedded_revision = os.environ.get("NEURORANKER_SOURCE_REVISION") or None
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    slurm = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
        and key
        in {
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_PARTITION",
            "SLURM_CPUS_PER_TASK",
            "SLURM_GPUS",
        }
    }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.sys.executable,
        "packages": packages,
        "git": {
            "embedded_revision": embedded_revision,
            "head": _run(["git", "rev-parse", "HEAD"], root),
            "status": _run(["git", "status", "--porcelain=v1"], root),
            "remote": _run(["git", "remote", "get-url", "origin"], root),
        },
        "locks": {
            "requirements-ci.lock": _sha256_optional(root / "requirements-ci.lock"),
            "requirements-runtime.lock": _sha256_optional(root / "requirements-runtime.lock"),
        },
        "gpu": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "slurm": slurm,
        "environment": {
            "LOADEDMODULES": os.environ.get("LOADEDMODULES", ""),
            "NEURORANKER_SOURCE_REVISION": os.environ.get(
                "NEURORANKER_SOURCE_REVISION", ""
            ),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        },
    }


def write_provenance(repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = collect_provenance(repo_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
