"""Resident whisper-server backend for live whisper.cpp transcription."""
from __future__ import annotations

import difflib
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..audio_capture import AudioChunk
from .audio_utils import has_enough_signal, pcm16_to_mono_float, resample, to_int16_pcm
from .whispercpp_common import (
    build_transcription_meta,
    join_segment_text,
    normalize_whispercpp_language,
    parse_server_segments,
    write_mono_wav,
)


@dataclass
class WhisperCppQualityGate:
    no_speech_threshold: float = 0.85
    avg_logprob_min: float = -1.2
    repetition_similarity: float = 0.92
    boilerplate_phrases: tuple[str, ...] = (
        "请不吝点赞",
        "訂閱",
        "订阅",
        "轉發",
        "转发",
        "打賞",
        "打赏",
    )
    dropped_reasons: list[str] = field(default_factory=list)

    def filter_segments(self, segments: list[dict[str, object]]) -> list[dict[str, object]]:
        self.dropped_reasons.clear()
        kept: list[dict[str, object]] = []
        previous_text = ""
        for segment in segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                self.dropped_reasons.append("empty")
                continue
            if self._is_low_quality(segment):
                self.dropped_reasons.append("low-quality")
                continue
            if self._is_boilerplate(text):
                self.dropped_reasons.append("boilerplate")
                continue
            if previous_text and self._similarity(previous_text, text) >= self.repetition_similarity:
                self.dropped_reasons.append("repetition")
                continue
            kept.append(segment)
            previous_text = text
        return kept

    def _is_low_quality(self, segment: dict[str, object]) -> bool:
        no_speech = _optional_float(segment.get("no_speech_prob"))
        avg_logprob = _optional_float(segment.get("avg_logprob"))
        if no_speech is not None and no_speech >= self.no_speech_threshold:
            return True
        if avg_logprob is not None and avg_logprob < self.avg_logprob_min:
            return True
        return False

    def _is_boilerplate(self, text: str) -> bool:
        compact = "".join(text.split())
        return any(phrase and phrase in compact for phrase in self.boilerplate_phrases)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_compact = "".join(left.split()).lower()
        right_compact = "".join(right.split()).lower()
        if not left_compact or not right_compact:
            return 0.0
        return difflib.SequenceMatcher(None, left_compact, right_compact).ratio()


class WhisperCppServerClient:
    def __init__(self, *, host: str, port: int, timeout_seconds: float = 30.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_seconds = max(0.5, float(timeout_seconds or 30.0))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ready(self) -> bool:
        request = urllib.request.Request(self.base_url + "/", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                return int(getattr(response, "status", 200) or 200) < 500
        except urllib.error.HTTPError as exc:
            return int(exc.code) < 500
        except Exception:
            return False

    def infer_wav(self, wav_path: Path, *, language: str) -> dict:
        boundary = f"----voice2text-{time.time_ns()}"
        fields = {
            "response_format": "verbose_json",
            "temperature": "0.0",
            "language": language,
        }
        body = self._multipart_body(boundary, fields, wav_path)
        request = urllib.request.Request(
            self.base_url + "/inference",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise RuntimeError(f"whisper-server returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("whisper-server JSON output has unexpected shape")
        return payload

    @staticmethod
    def _multipart_body(boundary: str, fields: dict[str, str], wav_path: Path) -> bytes:
        chunks: list[bytes] = []
        for key, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("ascii"),
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("ascii"),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                b'Content-Disposition: form-data; name="file"; filename="input.wav"\r\n',
                b"Content-Type: audio/wav\r\n\r\n",
                wav_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode("ascii"),
            ]
        )
        return b"".join(chunks)


class WhisperCppServerManager:
    def __init__(
        self,
        *,
        server_path: str | Path,
        model_path: str | Path,
        device: str = "vulkan",
        cpu_threads: int = 0,
        beam_size: int = 5,
        language: str = "auto",
        use_vad: bool = False,
        vad_model_path: str | Path | None = None,
        max_len: int = 0,
        host: str = "127.0.0.1",
        request_timeout_seconds: float = 30.0,
        progress_callback: Callable[[str], None] | None = None,
        popen_factory: Callable[..., subprocess.Popen] | None = None,
        client_factory: Callable[..., WhisperCppServerClient] | None = None,
    ) -> None:
        self.server_path = Path(server_path)
        self.model_path = Path(model_path)
        self.device = "cpu" if str(device or "").strip().lower().startswith("cpu") else "vulkan"
        self.cpu_threads = max(0, int(cpu_threads or 0))
        self.beam_size = max(1, int(beam_size or 5))
        self.language = normalize_whispercpp_language(language)
        self.use_vad = bool(use_vad)
        self.vad_model_path = Path(vad_model_path) if vad_model_path else None
        self.max_len = max(0, int(max_len or 0))
        self.host = host
        self.request_timeout_seconds = max(0.5, float(request_timeout_seconds or 30.0))
        self._progress_callback = progress_callback
        self._popen_factory = popen_factory or subprocess.Popen
        self._client_factory = client_factory or WhisperCppServerClient
        self._process: subprocess.Popen | None = None
        self._client: WhisperCppServerClient | None = None
        self._restart_attempted = False
        self._warm = False
        if not self.server_path.exists():
            raise RuntimeError(
                "whisper.cpp server binary not found: "
                f"{self.server_path}. Build it with app/build_whispercpp.ps1 or set VOICE2TEXT_WHISPERCPP_SERVER_BIN."
            )
        if not self.model_path.exists():
            raise RuntimeError(f"whisper.cpp ggml model not found: {self.model_path}")
        if self.use_vad:
            if self.vad_model_path is None:
                raise RuntimeError("whisper.cpp server VAD is enabled but no VAD model path was resolved.")
            if not self.vad_model_path.exists():
                raise RuntimeError(f"whisper.cpp VAD model not found: {self.vad_model_path}")

    @property
    def client(self) -> WhisperCppServerClient:
        if self._client is None:
            raise RuntimeError("whisper-server is not started")
        return self._client

    @property
    def enabled(self) -> bool:
        return self._warm and self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        port = _find_free_port(self.host)
        cmd = self._build_command(port)
        self._emit(f"whisper.cpp server starting: {self.host}:{port} ({self.device})")
        self._process = self._popen_factory(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.server_path.parent),
        )
        self._client = self._client_factory(host=self.host, port=port, timeout_seconds=self.request_timeout_seconds)
        self._wait_ready()

    def prewarm(self, language: str | None = None) -> None:
        self.start()
        if self.use_vad:
            self._warm = True
            self._emit("whisper.cpp server warmup inference skipped because server VAD can crash on zero-speech audio.")
            return
        lang = normalize_whispercpp_language(language) if language else self.language
        with tempfile.TemporaryDirectory(prefix="voice2text_whispercpp_server_warm_") as raw_tmp:
            wav_path = Path(raw_tmp) / "warmup.wav"
            write_mono_wav(wav_path, b"\x00\x00" * 16000)
            self._emit("whisper.cpp server warmup inference started.")
            self.client.infer_wav(wav_path, language=lang)
        self._warm = True
        self._emit("whisper.cpp server warmup completed.")

    def restart_once(self) -> bool:
        if self._restart_attempted:
            return False
        self._restart_attempted = True
        self.shutdown()
        try:
            self.prewarm(self.language)
            return True
        except Exception as exc:
            self._emit(f"whisper.cpp server restart failed: {exc}")
            return False

    def shutdown(self) -> None:
        proc = self._process
        self._process = None
        self._client = None
        self._warm = False
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3.0)
        except Exception:
            pass

    def _wait_ready(self, timeout_seconds: float = 30.0) -> None:
        deadline = time.perf_counter() + max(1.0, timeout_seconds)
        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"whisper-server exited early (exit={self._process.returncode})")
            if self._client is not None and self._client.ready():
                return
            time.sleep(0.1)
        raise RuntimeError("whisper-server readiness timed out")

    def _build_command(self, port: int) -> list[str]:
        cmd = [
            str(self.server_path),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(port),
            "-l",
            self.language,
            "-bs",
            str(self.beam_size),
        ]
        if self.cpu_threads > 0:
            cmd.extend(["-t", str(self.cpu_threads)])
        if self.device == "cpu":
            cmd.append("-ng")
        if self.use_vad:
            cmd.extend(["--vad", "--vad-model", str(self.vad_model_path)])
        if self.max_len > 0:
            cmd.extend(["--max-len", str(self.max_len)])
        return cmd

    def _emit(self, message: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:
            pass


class WhisperCppServerTranscriber:
    def __init__(
        self,
        *,
        manager: WhisperCppServerManager,
        fallback_transcriber,
        quality_gate: WhisperCppQualityGate | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._manager = manager
        self._fallback = fallback_transcriber
        self._quality_gate = quality_gate or WhisperCppQualityGate()
        self._progress_callback = progress_callback
        self._last_transcription_meta: dict[str, object] = {}

    def has_enough_signal(self, chunk: AudioChunk, threshold: float = 0.008, channel_mode: str = "mono") -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def prewarm(self, language: Optional[str] = None) -> None:
        self._manager.prewarm(language)

    def transcribe(self, chunk: AudioChunk, language: Optional[str] = None, channel_mode: str = "mono") -> str:
        started_at = time.perf_counter()
        timing: dict[str, object] = {
            "backend": "server",
            "input_sample_rate": int(getattr(chunk, "sample_rate", 0) or 0),
            "input_channels": int(getattr(chunk, "channels", 0) or 0),
            "alignment_enabled": False,
            "device": self._manager.device,
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
        lang = normalize_whispercpp_language(language)
        try:
            payload = self._request_server(audio, lang)
        except Exception as exc:
            if self._manager.restart_once():
                try:
                    payload = self._request_server(audio, lang)
                except Exception as retry_exc:
                    return self._fallback_transcribe(chunk, language, channel_mode, f"{exc}; retry failed: {retry_exc}")
            else:
                return self._fallback_transcribe(chunk, language, channel_mode, str(exc))
        segments = parse_server_segments(payload)
        filtered = self._quality_gate.filter_segments(segments)
        timing["dropped_segment_count"] = len(segments) - len(filtered)
        if self._quality_gate.dropped_reasons:
            timing["dropped_segment_reasons"] = ",".join(self._quality_gate.dropped_reasons)
        timing["total_seconds"] = time.perf_counter() - started_at
        detected_language = str(payload.get("detected_language") or payload.get("language") or ("" if lang == "auto" else lang))
        self._last_transcription_meta = build_transcription_meta(
            provider_timing=timing,
            segments=filtered,
            detected_language=detected_language,
            language_probabilities=payload.get("language_probabilities"),
        )
        return join_segment_text(filtered)

    def get_last_transcription_meta(self) -> dict[str, object]:
        return dict(self._last_transcription_meta)

    def close(self) -> None:
        self._manager.shutdown()

    shutdown = close

    def _request_server(self, audio, language: str) -> dict:
        if not self._manager.enabled:
            if not self._manager.restart_once():
                raise RuntimeError("whisper.cpp server is unavailable and restart budget is exhausted")
        with tempfile.TemporaryDirectory(prefix="voice2text_whispercpp_server_") as raw_tmp:
            wav_path = Path(raw_tmp) / "input.wav"
            write_mono_wav(wav_path, to_int16_pcm(audio))
            return self._manager.client.infer_wav(wav_path, language=language)

    def _fallback_transcribe(self, chunk: AudioChunk, language: Optional[str], channel_mode: str, reason: str) -> str:
        self._emit(f"whisper.cpp server unavailable; falling back to subprocess path: {reason}")
        try:
            text = self._fallback.transcribe(chunk, language=language, channel_mode=channel_mode)
            get_meta = getattr(self._fallback, "get_last_transcription_meta", None)
            if callable(get_meta):
                self._last_transcription_meta = dict(get_meta())
                timing = self._last_transcription_meta.get("provider_timing")
                if isinstance(timing, dict):
                    timing["backend"] = "subprocess-fallback"
                    timing["fallback_reason"] = reason
            return text
        except Exception as exc:
            self._emit(f"whisper.cpp subprocess fallback failed: {exc}")
            self._last_transcription_meta = {}
            return ""

    def _set_empty_meta(self, timing: dict[str, object], started_at: float) -> None:
        timing["total_seconds"] = time.perf_counter() - started_at
        self._last_transcription_meta = build_transcription_meta(provider_timing=timing, segments=[])

    def _emit(self, message: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:
            pass


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None
