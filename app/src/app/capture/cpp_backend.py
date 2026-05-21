"""C++ capture bridge backend for loopback/app source modes."""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ..audio_capture import AudioCaptureBase, AudioChunk

if TYPE_CHECKING:
    from ..config import RuntimeConfig

_FRAME_MAGIC = b"V2TB"
_FRAME_HEADER_BYTES = 16
_HEALTH_CACHE: dict[str, tuple[bool, str]] = {}
_PROCESS_LOOPBACK_CACHE: dict[str, tuple[bool, str]] = {}
_HELP_TEXT_CACHE: dict[str, str] = {}


class CppBridgeCapture(AudioCaptureBase):
    def __init__(
        self,
        *,
        source_mode: str,
        app_names: list[str] | None = None,
        source_device_id: str = "",
        debug_segment_path: Path | None = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._source_mode = source_mode
        self._app_names = list(app_names or [])
        self._source_device_id = source_device_id.strip()
        self._on_error = on_error
        self._on_status = on_status
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=512)
        self._running = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._debug_segment_path = debug_segment_path
        self._debug_segment_window_seconds = 6.0
        self._debug_segment_last_write = 0.0
        self._debug_segment_buffer = bytearray()
        self.sample_rate = 16000
        self.channels = 1

    def start(self) -> None:
        if self._running.is_set():
            return
        bridge_exe = _resolve_capture_bridge_executable()
        if bridge_exe is None:
            raise RuntimeError(
                "C++ capture bridge executable not found. Run `app/native/audio_bridge/build_bridge.ps1` or set VOICE2TEXT_CPP_CAPTURE_BRIDGE."
            )
        cmd = [str(bridge_exe), "--source-mode", self._source_mode]
        if self._app_names:
            cmd.extend(["--source-apps", ",".join(self._app_names)])
        if self._source_device_id:
            cmd.extend(["--source-device-id", self._source_device_id])
        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            creationflags=creationflags,
        )
        if self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to open C++ capture bridge pipes.")
        self._running.set()
        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._emit_status(f"C++ capture bridge started: {' '.join(cmd)}")

    def stop(self) -> None:
        self._running.clear()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._stdout_thread and self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=2.0)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2.0)
        self._stdout_thread = None
        self._stderr_thread = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read_chunk(self, timeout: float = 0.2) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _stdout_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        while self._running.is_set():
            header = _read_exact(stream, _FRAME_HEADER_BYTES)
            if header is None:
                break
            if header[:4] != _FRAME_MAGIC:
                self._emit_error("C++ capture bridge frame magic mismatch.")
                break
            sample_rate = int.from_bytes(header[4:8], "little", signed=False)
            channels = int.from_bytes(header[8:12], "little", signed=False)
            payload_len = int.from_bytes(header[12:16], "little", signed=False)
            if payload_len <= 0:
                continue
            payload = _read_exact(stream, payload_len)
            if payload is None:
                break
            self.sample_rate = max(8000, sample_rate)
            self.channels = max(1, channels)
            self._debug_append_segment(payload, self.sample_rate, self.channels)
            try:
                self._queue.put(AudioChunk(payload, self.sample_rate, self.channels), timeout=0.2)
            except queue.Full:
                pass
        if self._running.is_set():
            return_code = proc.poll()
            self._emit_error(f"C++ capture bridge stdout ended unexpectedly (exit={return_code}).")
        self._running.clear()

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            if not self._running.is_set():
                break
            message = raw.decode("utf-8", errors="replace").strip()
            if message:
                self._emit_status(message)

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)

    def _emit_status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    def _debug_append_segment(self, pcm16: bytes, sample_rate: int, channels: int) -> None:
        path = self._debug_segment_path
        if path is None:
            return
        if not pcm16:
            return
        now = time.monotonic()
        frame_bytes = max(2, int(channels) * 2)
        max_bytes = int(max(1.0, self._debug_segment_window_seconds) * sample_rate * frame_bytes)
        self._debug_segment_buffer.extend(pcm16)
        if len(self._debug_segment_buffer) > max_bytes:
            overflow = len(self._debug_segment_buffer) - max_bytes
            trim = overflow - (overflow % frame_bytes)
            if trim > 0:
                del self._debug_segment_buffer[:trim]
        if now - self._debug_segment_last_write < 0.6:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(max(1, int(channels)))
                wf.setsampwidth(2)
                wf.setframerate(max(8000, int(sample_rate)))
                wf.writeframes(bytes(self._debug_segment_buffer))
            self._debug_segment_last_write = now
        except Exception:
            return


def build_cpp_capture_from_config(
    config: RuntimeConfig,
    *,
    on_error: Optional[Callable[[str], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> AudioCaptureBase | None:
    mode = (config.source_mode or "loopback").strip().lower()
    if mode not in {"loopback", "app"}:
        return None
    bridge_exe = _resolve_capture_bridge_executable()
    if bridge_exe is None:
        return None
    healthy, reason = _check_bridge_health(bridge_exe)
    if not healthy:
        if on_status is not None:
            on_status(
                "C++ capture bridge is unavailable on this runtime; using Python fallback backend. "
                f"reason={reason}"
            )
        return None
    if mode == "loopback" and list(config.source_device_indices):
        if on_status is not None:
            on_status(
                "C++ loopback backend currently captures default output endpoint. "
                "Numeric source-device index selection uses Python fallback backend."
            )
        return None
    if mode == "app":
        supports_process_loopback, reason = _check_process_loopback_support(bridge_exe)
        if not supports_process_loopback:
            if on_status is not None:
                on_status(
                    "C++ app-mode process loopback is unavailable in current bridge build; "
                    f"using Python fallback backend. reason={reason}"
                )
            return None
        app_names = list(config.source_app_names)
        if not app_names and config.source_app_name.strip():
            app_names = [config.source_app_name.strip()]
        normalized = _normalize_app_target_names(app_names)
        if not normalized:
            if on_status is not None:
                on_status("No app target selected for app mode; using Python fallback backend.")
            return None
        if list(config.source_device_indices) and on_status is not None:
            on_status("App mode will prioritize Application Loopback Capture by process name.")
        return CppBridgeCapture(
            source_mode="app",
            app_names=normalized,
            source_device_id="",
            debug_segment_path=_resolve_cpp_segment_debug_path(config),
            on_error=on_error,
            on_status=on_status,
        )
    return CppBridgeCapture(
        source_mode="loopback",
        app_names=[],
        source_device_id="",
        debug_segment_path=_resolve_cpp_segment_debug_path(config),
        on_error=on_error,
        on_status=on_status,
    )


def _resolve_capture_bridge_executable() -> Path | None:
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


def _read_exact(stream, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _check_bridge_health(exe_path: Path) -> tuple[bool, str]:
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
        # Accept 1 for stricter parser/build variants that still run correctly.
        result = (True, f"probe-exit={code}")
        _HEALTH_CACHE[key] = result
        return result

    # Windows loader/runtime crash codes may appear as negative or unsigned 32-bit.
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


def _check_process_loopback_support(exe_path: Path) -> tuple[bool, str]:
    key = str(exe_path.resolve())
    cached = _PROCESS_LOOPBACK_CACHE.get(key)
    if cached is not None:
        return cached
    help_text = _read_bridge_help_text(exe_path)
    if "--probe-process-loopback" not in help_text:
        result = (False, _legacy_probe_unavailable_reason(exe_path))
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


def _read_bridge_help_text(exe_path: Path) -> str:
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


def _legacy_probe_unavailable_reason(exe_path: Path) -> str:
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


def _resolve_cpp_segment_debug_path(config: RuntimeConfig) -> Path | None:
    if not bool(getattr(config, "debug_mode", False)):
        return None
    raw_log_dir = str(getattr(config, "log_dir", "logs") or "logs").strip()
    log_dir = Path(raw_log_dir) if raw_log_dir else Path("logs")
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parents[2] / log_dir
    log_dir = log_dir.resolve()
    segments_dir = log_dir.parent / "segments"
    return segments_dir / "latest_segment_cpp_bridge.wav"


def _normalize_app_target_names(values: list[str]) -> list[str]:
    names: list[str] = []
    for raw in values:
        normalized = _normalize_single_target_name(raw)
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def _normalize_single_target_name(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    exe_tokens = re.findall(r"([0-9A-Za-z_.-]+\.exe)", raw, flags=re.IGNORECASE)
    if exe_tokens:
        return exe_tokens[-1].strip().lower()
    for inner in re.findall(r"\(([^)]+)\)", raw):
        token = inner.strip().lower()
        if token:
            return token
    return raw.lower()
