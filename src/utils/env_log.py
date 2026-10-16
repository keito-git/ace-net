"""
Environment manifest for reproducibility audit.

Every training run calls log_environment() before any training step.
The resulting env_log.json accompanies all result files and checkpoints.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_commit() -> str:
    """Return current git HEAD hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _gpu_uuids() -> List[str]:
    """Return list of physical GPU UUIDs from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        return ["no_gpu"]
    except Exception:
        return ["no_gpu"]


def _file_hash(filepath: str, read_bytes: int = 1_048_576) -> str:
    """SHA-256 of the first `read_bytes` of a file (1 MB default for speed)."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            h.update(f.read(read_bytes))
        return h.hexdigest()[:16]
    except Exception:
        return "hash_error"


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not_installed"


def _cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "no_cuda"
    except ImportError:
        return "not_installed"


def log_environment(
    seed: int,
    dataset_name: str,
    output_dir: str,
    data_files: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an env_log manifest and save it to output_dir/env_log.json.

    Returns the manifest dict (also persisted to disk).
    """
    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "torch_version": _torch_version(),
        "cuda_version": _cuda_version(),
        "git_commit": _git_commit(),
        "gpu_uuids": _gpu_uuids(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "seed": seed,
        "dataset": dataset_name,
        "data_hashes": {f: _file_hash(f) for f in (data_files or [])},
    }
    if extra:
        manifest["extra"] = extra

    out_path = Path(output_dir) / "env_log.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[env_log] Manifest saved: {out_path}")
    return manifest
