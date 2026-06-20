"""Session recorder: a transparent capture decorator that saves the exact live
PCM to WAV + a redacted manifest, so a live session becomes a deterministic,
replayable artifact (round 0020). Replay reuses the existing file-replay path.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Callable, Optional
import wave

from ..audio_capture import AudioCaptureBase, AudioChunk

_SENSITIVE_CONFIG_KEYS = {"whisperx_hf_token"}
# Substrings that mark a key as a credential to redact (future translation/cloud backends may add
# `translation_*_api_key` / `*_token` / `*_secret` fields — they are redacted automatically).
_SENSITIVE_KEY_MARKERS = ("token", "api_key", "apikey", "secret", "password", "passwd")


def _is_sensitive_key(key: str) -> bool:
    if key in _SENSITIVE_CONFIG_KEYS:
        return True
    lowered = str(key).lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def redact_config_snapshot(snapshot: dict | None) -> dict:
    """Copy a config snapshot with sensitive values redacted (HF token, API keys, secrets)."""
    out: dict = {}
    for (key, value) in dict(snapshot or {}).items():
        if _is_sensitive_key(key):
            out[key] = "<redacted>" if str(value or "").strip() else ""
        else:
            out[key] = value
    return out


def default_recording_dir(base_dir: str | Path) -> Path:
    """A fresh `recordings/<YYYYmmdd_HHMMSS>/` directory under base_dir."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / "recordings" / stamp


# The STT-relevant config fields restored on replay so the run reproduces the
# recorded session's behaviour. Infra (log_dir, dirs, overlay, tokens) is left as
# the current environment's.
_REPLAY_CONFIG_FIELDS = (
    "model_size", "stt_model_path", "stt_variant", "compute_type", "whisper_beam_size",
    "whisper_batch_size", "segment_seconds", "hop_seconds", "overlap_merge_method",
    "runtime_preset", "source_language", "preprocess_enabled",
    "whisperx_enable_phoneme_asr", "whisperx_enable_forced_alignment", "whisperx_enable_vad",
    "whisperx_vad_method", "whisperx_enable_diarization", "whisperx_alignment_model",
    "whisperx_alignment_language", "whisperx_alignment_device", "whisperx_diarization_device",
    "whisperx_diarization_model", "whisperx_speaker_profile_enabled",
    "whisperx_speaker_profile_backend", "subtitle_display_script", "cjk_no_space_gap_seconds",
    "speaker_marker_style", "speaker_pause_break_seconds", "whisperx_rolling_prompt_chars",
)


def load_session_manifest(session_dir: str | Path) -> dict:
    """Load a recording's manifest.json; accepts the dir or the manifest path."""
    p = Path(session_dir).expanduser()
    manifest_path = (p / "manifest.json") if p.is_dir() else p
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    wav_name = str(manifest.get("wav") or "session.wav")
    manifest["_session_dir"] = str(manifest_path.parent)
    manifest["_wav_path"] = str(manifest_path.parent / wav_name)
    return manifest


def replay_config_overrides(manifest: dict) -> dict:
    """The recorded STT config subset to restore on replay (excludes infra/secrets)."""
    cfg = dict(manifest.get("config") or {})
    return {k: cfg[k] for k in _REPLAY_CONFIG_FIELDS if k in cfg}


def apply_replay_session(config, session_dir: str | Path) -> dict:
    """Point `config` at a recorded session: source_mode=file + recorded STT config.

    Returns the loaded manifest (with `_wav_path`). Raises on a missing/invalid
    recording so the caller can surface it.
    """
    manifest = load_session_manifest(session_dir)
    wav_path = manifest["_wav_path"]
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"Recorded WAV not found: {wav_path}")
    config.source_mode = "file"
    config.source_file_path = wav_path
    config.source_file_replay_speed = 0.0
    for (key, value) in replay_config_overrides(manifest).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return manifest


class RecordingAudioCapture(AudioCaptureBase):
    """Wrap any `AudioCaptureBase`; record what it yields, forward it unchanged."""

    def __init__(
        self,
        inner: AudioCaptureBase,
        *,
        out_dir: str | Path,
        config_snapshot: dict | None = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._inner = inner
        self._out_dir = Path(out_dir)
        self._config_snapshot = dict(config_snapshot or {})
        self._on_status = on_status
        self.sample_rate = int(getattr(inner, "sample_rate", 16000) or 16000)
        self.channels = int(getattr(inner, "channels", 1) or 1)
        self._wav: Optional[wave.Wave_write] = None
        self._wav_params: Optional[tuple[int, int]] = None
        self._chunks: list[tuple[float, int]] = []  # (arrival_ms, byte_count)
        self._total_bytes = 0
        self._started_at = 0.0
        self._finalized = False
        self._wav_path = self._out_dir / "session.wav"
        self._manifest_path = self._out_dir / "manifest.json"

    def __getattr__(self, name: str):
        # Transparent: delegate anything not defined here to the wrapped capture
        # (e.g. duration_seconds, backend-specific helpers). Guarded against the
        # __init__-before-_inner recursion.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    def start(self) -> None:
        self._inner.start()
        self.sample_rate = int(getattr(self._inner, "sample_rate", self.sample_rate) or self.sample_rate)
        self.channels = int(getattr(self._inner, "channels", self.channels) or self.channels)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.monotonic()
        self._emit(f"Session recording started: {self._out_dir}")

    def read_chunk(self, timeout: float = 0.2) -> Optional[AudioChunk]:
        chunk = self._inner.read_chunk(timeout)
        if chunk is not None and getattr(chunk, "pcm16", b""):
            self._record(chunk)
        return chunk

    def stop(self) -> None:
        try:
            self._inner.stop()
        finally:
            self._finalize()

    def _record(self, chunk: AudioChunk) -> None:
        try:
            if self._wav is None:
                self._open_wav(int(chunk.sample_rate), int(chunk.channels))
            self._wav.writeframes(chunk.pcm16)
            self._total_bytes += len(chunk.pcm16)
            self._chunks.append((round((time.monotonic() - self._started_at) * 1000.0, 1), len(chunk.pcm16)))
        except Exception as exc:  # never let recording break capture
            self._emit(f"Session recording write failed: {exc}")

    def _open_wav(self, sample_rate: int, channels: int) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        wav = wave.open(str(self._wav_path), "wb")
        wav.setnchannels(max(1, int(channels)))
        wav.setsampwidth(2)
        wav.setframerate(max(8000, int(sample_rate)))
        self._wav = wav
        self._wav_params = (int(sample_rate), int(channels))

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
        self._write_manifest()
        self._emit(
            f"Session recording saved: {self._out_dir} "
            f"({len(self._chunks)} chunks, {self._total_bytes} bytes)"
        )

    def _write_manifest(self) -> None:
        (sr, ch) = self._wav_params or (self.sample_rate, self.channels)
        duration = self._total_bytes / float(max(1, sr * max(1, ch) * 2))
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wav": self._wav_path.name,
            "sample_rate": int(sr),
            "channels": int(ch),
            "sample_width_bytes": 2,
            "chunk_count": int(len(self._chunks)),
            "total_pcm_bytes": int(self._total_bytes),
            "duration_seconds": round(float(duration), 3),
            "chunks": [[t, b] for (t, b) in self._chunks],
            "config": redact_config_snapshot(self._config_snapshot),
        }
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            self._emit(f"Session recording manifest write failed: {exc}")

    def _emit(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)
