"""Bridge executable discovery and capability probe helpers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

_HEALTH_CACHE: dict[str, tuple[bool, str]] = {}
_PROCESS_LOOPBACK_CACHE: dict[str, tuple[bool, str]] = {}
_HELP_TEXT_CACHE: dict[str, str] = {}


def resolve_capture_bridge_executable() -> Path | None:
    env = os.environ.get("VOICE2TEXT_CPP_CAPTURE_BRIDGE", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "app" / "src" / "runtime_bin" / "voice2text_capture_bridge.exe",
        repo_root / "app" / "native" / "audio_bridge" / "build" / "Release" / "voice2text_capture_bridge.exe",
        repo_root / "app" / "native" / "audio_bridge" / "build" / "voice2text_capture_bridge.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def check_bridge_health(exe_path: Path) -> tuple[bool, str]:
    key = str(exe_path.resolve())
    cached = _HEALTH_CACHE.get(key)
    if cached is not None:
        return cached
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        proc = subprocess.run(
            [str(exe_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=4.0,
            creationflags=creationflags,
            check=False,
        )
    except Exception as exc:
        result = (False, f"probe-failed: {exc}")
        _HEALTH_CACHE[key] = result
        return result
    code = int(proc.returncode)
    if code in (0, 1):
        result = (True, f"probe-exit={code}")
        _HEALTH_CACHE[key] = result
        return result
    if code < 0 or code >= 0x80000000:
        signed = code
        if code >= 0x80000000:
            signed = code - 0x100000000
        result = (
            False,
            f"probe-crashed exit={signed} (0x{code:08X}); likely missing/incompatible runtime DLL dependencies for bridge executable",
        )
        _HEALTH_CACHE[key] = result
        return result
    result = (False, f"probe-exit={code}")
    _HEALTH_CACHE[key] = result
    return result


def check_process_loopback_support(exe_path: Path) -> tuple[bool, str]:
    key = str(exe_path.resolve())
    cached = _PROCESS_LOOPBACK_CACHE.get(key)
    if cached is not None:
        return cached
    help_text = read_bridge_help_text(exe_path)
    if "--probe-process-loopback" not in help_text:
        result = (False, legacy_probe_unavailable_reason(exe_path))
        _PROCESS_LOOPBACK_CACHE[key] = result
        return result
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        proc = subprocess.run(
            [str(exe_path), "--probe-process-loopback"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=3.0,
            creationflags=creationflags,
            check=False,
        )
    except Exception as exc:
        result = (False, f"probe-failed: {exc}")
        _PROCESS_LOOPBACK_CACHE[key] = result
        return result
    code = int(proc.returncode)
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
    if code == 0:
        result = (True, stderr_text or "supported")
        _PROCESS_LOOPBACK_CACHE[key] = result
        return result
    result = (False, stderr_text or f"unsupported-exit={code}")
    _PROCESS_LOOPBACK_CACHE[key] = result
    return result


def read_bridge_help_text(exe_path: Path) -> str:
    key = str(exe_path.resolve())
    cached = _HELP_TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = ""
    try:
        proc = subprocess.run(
            [str(exe_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=4.0,
            creationflags=creationflags,
            check=False,
        )
        stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
        text = f"{stdout_text}\n{stderr_text}".strip()
    except Exception:
        text = ""
    _HELP_TEXT_CACHE[key] = text
    return text


def legacy_probe_unavailable_reason(exe_path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[4]
    bridge_main = repo_root / "app" / "native" / "audio_bridge" / "src" / "main.cpp"
    try:
        if bridge_main.exists() and bridge_main.stat().st_mtime > exe_path.stat().st_mtime:
            return (
                "bridge executable is older than source and does not expose "
                "--probe-process-loopback; rebuild/deploy bridge binary"
            )
    except Exception:
        pass
    return "bridge executable does not expose --probe-process-loopback"
