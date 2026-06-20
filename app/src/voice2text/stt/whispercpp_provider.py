"""whisper.cpp subprocess STT provider with segment-span synthesized timestamps."""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from ..audio_capture import AudioChunk
from .audio_utils import has_enough_signal, pcm16_to_mono_float, resample, to_int16_pcm
from .whispercpp_common import (
    build_transcription_meta,
    join_segment_text,
    normalize_whispercpp_language,
    parse_cli_segments,
    read_json_object,
    synthesize_segment_word_timestamps,
    write_mono_wav,
)

_VULKAN_FAILURE_RE = re.compile(r"(vulkan|ggml_vulkan|vk_|no device|device.*not.*found|failed.*gpu)", re.IGNORECASE)


class WhisperCppTranscriber:
    def __init__(
        self,
        *,
        binary_path: str | Path,
        model_path: str | Path,
        device: str = "vulkan",
        cpu_threads: int = 0,
        beam_size: int = 5,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._binary_path = Path(binary_path)
        self._model_path = Path(model_path)
        self._device = "cpu" if str(device or "").strip().lower().startswith("cpu") else "vulkan"
        self._cpu_threads = max(0, int(cpu_threads or 0))
        self._beam_size = max(1, int(beam_size or 5))
        self._progress_callback = progress_callback
        self._last_transcription_meta: dict[str, object] = {}
        if not self._binary_path.exists():
            raise RuntimeError(
                "whisper.cpp backend binary not found: "
                f"{self._binary_path}. Build it with app/build_whispercpp.ps1 or set VOICE2TEXT_WHISPERCPP_BIN."
            )
        if not self._model_path.exists():
            raise RuntimeError(
                "whisper.cpp ggml model not found: "
                f"{self._model_path}. Set stt_whispercpp_model_path or enable stt_auto_download."
            )

    def has_enough_signal(self, chunk: AudioChunk, threshold: float = 0.008, channel_mode: str = "mono") -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk: AudioChunk, language: Optional[str] = None, channel_mode: str = "mono") -> str:
        started_at = time.perf_counter()
        timing: dict[str, object] = {
            "input_sample_rate": int(getattr(chunk, "sample_rate", 0) or 0),
            "input_channels": int(getattr(chunk, "channels", 0) or 0),
            "alignment_enabled": False,
            "device": self._device,
        }
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            self._set_empty_meta(timing, started_at)
            return ""
        if int(chunk.sample_rate) != 16000:
            audio = resample(audio, int(chunk.sample_rate), 16000)
        if audio.size == 0:
            self._set_empty_meta(timing, started_at)
            return ""
        timing["audio_samples"] = int(audio.size)
        timing["audio_seconds"] = float(audio.size) / 16000.0
        lang = self._normalize_cli_language(language)
        with tempfile.TemporaryDirectory(prefix="voice2text_whispercpp_") as tmp:
            tmp_dir = Path(tmp)
            wav_path = tmp_dir / "input.wav"
            out_prefix = tmp_dir / "out"
            write_mono_wav(wav_path, to_int16_pcm(audio))
            result = self._run_cli(wav_path, out_prefix, lang, use_cpu=(self._device == "cpu"))
            if result.returncode != 0 and self._device != "cpu" and _VULKAN_FAILURE_RE.search(result.stderr or result.stdout or ""):
                self._emit("whisper.cpp Vulkan unavailable; retrying with CPU (-ng).")
                timing["vulkan_fallback"] = "cpu"
                result = self._run_cli(wav_path, out_prefix, lang, use_cpu=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"whisper.cpp transcription failed (exit={result.returncode}): {detail}")
            payload = read_json_object(out_prefix.with_suffix(".json"), label="whisper.cpp")
        segments = parse_cli_segments(payload)
        text = join_segment_text(segments)
        timing["total_seconds"] = time.perf_counter() - started_at
        self._last_transcription_meta = build_transcription_meta(
            provider_timing=timing,
            segments=segments,
            detected_language="" if lang == "auto" else lang,
        )
        return text

    def get_last_transcription_meta(self) -> dict[str, object]:
        return dict(self._last_transcription_meta)

    def _run_cli(self, wav_path: Path, out_prefix: Path, lang: str, *, use_cpu: bool) -> subprocess.CompletedProcess[str]:
        cmd = [
            str(self._binary_path),
            "-m",
            str(self._model_path),
            "-f",
            str(wav_path),
            "-l",
            lang,
            "-bs",
            str(self._beam_size),
            "-oj",
            "-np",
            "-of",
            str(out_prefix),
        ]
        if self._cpu_threads > 0:
            cmd.extend(["-t", str(self._cpu_threads)])
        if use_cpu:
            cmd.append("-ng")
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=str(self._binary_path.parent),
        )

    @staticmethod
    def _synthesize_segment_word_timestamps(segment: dict[str, object]) -> list[dict[str, object]]:
        return synthesize_segment_word_timestamps(segment)

    @staticmethod
    def _normalize_cli_language(language: Optional[str]) -> str:
        return normalize_whispercpp_language(language)

    def _set_empty_meta(self, timing: dict[str, object], started_at: float) -> None:
        timing["total_seconds"] = time.perf_counter() - started_at
        self._last_transcription_meta = {
            "provider": "whispercpp",
            "stability_ratio": 1.0,
            "token_count": 0,
            "stable_token_count": 0,
            "alignment_enabled": False,
            "token_timestamps": [],
            "detected_language": "",
            "speaker_turns": [],
            "speaker_turn_count": 0,
            "provider_timing": timing,
        }

    def _emit(self, message: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:
            pass
