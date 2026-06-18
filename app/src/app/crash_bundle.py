"""One-shot diagnostics bundle: gather recent logs/traces/settings + an environment report into a
single redacted zip for bug reports (round 0025).

Headless, best-effort, and safe to run when the app is already broken: every probe and file copy is
guarded, missing inputs are skipped (not fatal), and the HF token is never written into the bundle.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Callable
import zipfile

from .capture.session_recorder import redact_config_snapshot

# Skip individual files larger than this so a runaway log can't bloat the bundle.
_MAX_FILE_BYTES = 25 * 1024 * 1024


def _safe(fn: Callable[[], object], fallback: object = None) -> object:
    try:
        return fn()
    except Exception as exc:  # diagnostics must never throw
        return f"<error: {exc}>" if fallback is None else fallback


def _git_revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
        return (out.stdout or b"").decode("utf-8", errors="replace").strip() or "<unknown>"
    except Exception:
        return "<unknown>"


def _torch_cuda_report() -> dict[str, object]:
    try:
        import torch  # type: ignore

        return {
            "torch_version": str(getattr(torch, "__version__", "?")),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {"torch_version": f"<unavailable: {exc}>", "cuda_available": False, "cuda_device_count": 0}


def collect_environment(config: object | None = None) -> dict[str, object]:
    """Best-effort environment report (platform / torch / ffmpeg / bridge / cache / git)."""
    from .config import RuntimeConfig
    from .stt import healthcheck
    from .stt.model_cache import cache_summary

    cfg = config if config is not None else RuntimeConfig()
    env: dict[str, object] = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "platform": _safe(lambda: platform.platform()),
        "python_version": sys.version.split()[0],
        "cpu_count": _safe(lambda: os.cpu_count(), 0),
        "app_git_revision": _git_revision(),
    }
    env.update(_torch_cuda_report())
    env["ffmpeg"] = _safe(lambda: healthcheck.check_ffmpeg(cfg).as_dict())
    env["capture_bridge"] = _safe(lambda: healthcheck.check_capture_bridge().as_dict())
    env["model_cache"] = _safe(lambda: cache_summary())
    return env


def _collect_recent(directory: Path, pattern: str, limit: int) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        files = [p for p in directory.glob(pattern) if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: _safe(lambda: p.stat().st_mtime, 0.0), reverse=True)
    return files[: max(0, int(limit))]


def _redacted_runtime_settings(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        data = redact_config_snapshot(data)
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def create_crash_bundle(
    config: object | None = None,
    *,
    out_dir: str | Path | None = None,
    reason: str = "",
    recent_per_pattern: int = 5,
    on_status: Callable[[str], None] | None = None,
) -> Path:
    """Write a redacted diagnostics zip and return its path. Never raises on missing inputs."""
    from .config import RuntimeConfig

    cfg = config if config is not None else RuntimeConfig()
    log_dir = Path(getattr(cfg, "log_dir", "logs") or "logs").resolve()
    src_root = log_dir.parent
    debug_logs = src_root / "debug_logs"
    runtime_settings = src_root / "runtime_settings.json"

    bundle_dir = Path(out_dir).resolve() if out_dir is not None else (src_root / "crash_bundles")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = bundle_dir / f"crash_{stamp}.zip"

    included: list[str] = []
    skipped: list[str] = []

    def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
        try:
            size = src.stat().st_size
        except OSError:
            skipped.append(f"{arcname} (stat failed)")
            return
        if size > _MAX_FILE_BYTES:
            skipped.append(f"{arcname} (too large: {size} bytes)")
            return
        try:
            zf.write(src, arcname)
            included.append(arcname)
        except OSError as exc:
            skipped.append(f"{arcname} ({exc})")

    config_snapshot = _safe(lambda: redact_config_snapshot(dataclasses.asdict(cfg)), {})
    environment = collect_environment(cfg)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in _collect_recent(log_dir, "*.log", recent_per_pattern):
            _add(zf, src, f"logs/{src.name}")
        crash_trace = log_dir / "python_crash_trace.log"
        if crash_trace.is_file() and f"logs/{crash_trace.name}" not in included:
            _add(zf, crash_trace, f"logs/{crash_trace.name}")
        for src in _collect_recent(debug_logs, "debug_trace_*.jsonl", recent_per_pattern):
            _add(zf, src, f"debug_logs/{src.name}")

        redacted_settings = _redacted_runtime_settings(runtime_settings)
        if redacted_settings is not None:
            zf.writestr("runtime_settings.json", redacted_settings)
            included.append("runtime_settings.json")
        else:
            skipped.append("runtime_settings.json (missing/unreadable)")

        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "reason": str(reason or ""),
            "environment": environment,
            "config": config_snapshot,
            "included_files": list(included),
            "skipped": list(skipped),
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    if on_status is not None:
        on_status(f"Crash bundle written: {zip_path} ({len(included)} files)")
    return zip_path
