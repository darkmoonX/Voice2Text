"""Persistent verdict cache for the isolated WhisperX CUDA alignment probe."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def collect_probe_signature(*, align_model_repo: str) -> dict[str, str]:
    """Build the cache signature without importing heavy runtimes at module import time."""
    torch_version = "unknown"
    cuda_version = "unknown"
    gpu_name = "unknown"
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "") or "unknown")
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "none")
        try:
            if torch.cuda.is_available():
                gpu_name = str(torch.cuda.get_device_name(0) or "unknown")
            else:
                gpu_name = "cuda-unavailable"
        except Exception:
            gpu_name = "cuda-query-failed"
    except Exception:
        pass
    return {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "align_model_repo": str(align_model_repo or ""),
    }


def cache_key(signature: dict[str, str]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_cached_verdict(cache_path: Path, signature: dict[str, str]) -> bool | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return None
    entry = entries.get(cache_key(signature))
    if not isinstance(entry, dict):
        return None
    cached_sig = entry.get("signature")
    if cached_sig != signature:
        return None
    verdict = entry.get("cuda_safe")
    if isinstance(verdict, bool):
        return verdict
    return None


def write_cached_verdict(cache_path: Path, signature: dict[str, str], *, cuda_safe: bool, reason: str = "") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    entries[cache_key(signature)] = {
        "signature": dict(signature),
        "cuda_safe": bool(cuda_safe),
        "reason": str(reason or ""),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data["version"] = 1
    data["entries"] = entries
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(cache_path)
