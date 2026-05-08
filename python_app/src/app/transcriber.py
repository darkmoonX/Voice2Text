from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
try:
    from faster_whisper.utils import download_model as fw_download_model
except Exception:
    fw_download_model = None

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None

from .audio_capture import AudioChunk
from .model_paths import library_model_dir


class FasterWhisperTranscriber:
    _opencc_hant = None
    _opencc_hans = None

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        target_sample_rate: int = 16000,
        max_context: Optional[int] = None,
        entropy_thold: Optional[float] = None,
        logprob_thold: Optional[float] = None,
        no_speech_thold: Optional[float] = None,
        temperature: Optional[float] = None,
        beam_size: Optional[int] = None,
        best_of: Optional[int] = None,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._model_dir = library_model_dir("faster-whisper")
        self._max_context = max_context if max_context and max_context > 0 else None
        self._entropy_thold = entropy_thold
        self._logprob_thold = logprob_thold
        self._no_speech_thold = no_speech_thold
        self._temperature = 0.0 if temperature is None else float(temperature)
        self._beam_size = max(1, int(beam_size)) if beam_size is not None else 1
        self._best_of = max(1, int(best_of)) if best_of is not None else 1

        model_ref = model_size.strip() or "small"
        model_path = Path(model_ref)
        model_arg = self._resolve_model_arg(model_ref, model_path)

        self._model = WhisperModel(
            model_arg,
            device=device,
            compute_type=compute_type,
            download_root=str(self._model_dir),
        )

    def _resolve_model_arg(self, model_ref: str, model_path: Path) -> str:
        if model_path.exists():
            return str(model_path)

        # Keep Hugging Face repo IDs unchanged; we enforce local named folders for built-in aliases.
        if "/" in model_ref or "\\" in model_ref:
            return model_ref

        target_dir = self._model_dir / model_ref
        if self._is_model_dir_ready(target_dir):
            return str(target_dir)

        legacy_snapshot = self._find_legacy_snapshot_dir(model_ref)
        if legacy_snapshot is not None:
            self._materialize_named_model_dir(target_dir, legacy_snapshot)
            if self._is_model_dir_ready(target_dir):
                return str(target_dir)

        if fw_download_model is not None:
            try:
                fw_download_model(model_ref, output_dir=str(target_dir))
                if self._is_model_dir_ready(target_dir):
                    return str(target_dir)
            except Exception:
                pass

        return model_ref

    @staticmethod
    def _is_model_dir_ready(path: Path) -> bool:
        return path.is_dir() and (path / "model.bin").exists() and (path / "config.json").exists()

    def _find_legacy_snapshot_dir(self, model_ref: str) -> Optional[Path]:
        normalized_ref = model_ref.strip().lower().replace("_", "-").replace("/", "-")
        if not normalized_ref:
            return None

        candidates: list[Path] = []
        for cache_dir in self._model_dir.glob("models--*"):
            if not cache_dir.is_dir():
                continue
            if normalized_ref not in cache_dir.name.lower():
                continue

            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot in snapshots_dir.iterdir():
                    if self._is_model_dir_ready(snapshot):
                        candidates.append(snapshot)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return candidates[0]

    def _materialize_named_model_dir(self, target_dir: Path, source_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        required_files = {
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
        }

        for file_path in source_dir.iterdir():
            if not file_path.is_file():
                continue

            file_name = file_path.name
            if file_name not in required_files and not file_name.startswith("vocabulary"):
                continue

            target_file = target_dir / file_name
            if target_file.exists():
                continue

            try:
                os.link(file_path, target_file)
            except OSError:
                shutil.copy2(file_path, target_file)

    def has_enough_signal(
        self,
        chunk: AudioChunk,
        threshold: float = 0.008,
        channel_mode: str = "mono",
    ) -> bool:
        audio = self._pcm16_to_mono_float(
            chunk.pcm16,
            chunk.channels,
            channel_mode=channel_mode,
        )
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms >= threshold

    def transcribe(
        self,
        chunk: AudioChunk,
        language: Optional[str] = None,
        channel_mode: str = "mono",
    ) -> str:
        audio = self._pcm16_to_mono_float(
            chunk.pcm16,
            chunk.channels,
            channel_mode=channel_mode,
        )
        if audio.size == 0:
            return ""

        audio = self._resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 4:
            return ""

        whisper_language, zh_script = self._normalize_language_hint(language)
        text = self._transcribe_with_long_window_splitting(audio, language=whisper_language)
        return self._normalize_chinese_script(text, zh_script)

    def _transcribe_with_long_window_splitting(
        self,
        audio: np.ndarray,
        language: Optional[str],
    ) -> str:
        # Split long windows into overlapping passes to avoid instability on large segments.
        max_window_samples = int(self._target_sample_rate * 4.8)
        if audio.size <= max_window_samples:
            return self._transcribe_single(audio, language=language)

        step_samples = int(self._target_sample_rate * 3.6)
        if step_samples <= 0:
            step_samples = max_window_samples

        parts: list[str] = []
        for start in range(0, int(audio.size), step_samples):
            end = min(start + max_window_samples, int(audio.size))
            if end <= start:
                break

            piece = self._transcribe_single(audio[start:end], language=language)
            if piece:
                parts.append(piece)

            if end >= int(audio.size):
                break

        return self._merge_text_parts(parts)

    def _transcribe_single(self, audio: np.ndarray, language: Optional[str]) -> str:
        kwargs: dict[str, object] = {
            "language": language,
            "beam_size": self._beam_size,
            "best_of": self._best_of,
            "temperature": self._temperature,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if self._max_context is not None:
            kwargs["max_new_tokens"] = self._max_context
        if self._entropy_thold is not None:
            kwargs["compression_ratio_threshold"] = self._entropy_thold
        if self._logprob_thold is not None:
            kwargs["log_prob_threshold"] = self._logprob_thold
        if self._no_speech_thold is not None:
            kwargs["no_speech_threshold"] = self._no_speech_thold

        segments, _ = self._model.transcribe(audio, **kwargs)

        texts: list[str] = []
        for seg in segments:
            cleaned = seg.text.strip()
            if cleaned:
                texts.append(cleaned)

        if not texts:
            return ""

        return " ".join(texts).replace("  ", " ").strip()

    @staticmethod
    def _merge_text_parts(parts: list[str]) -> str:
        merged = ""
        for piece in parts:
            cleaned = piece.strip()
            if not cleaned:
                continue

            if not merged:
                merged = cleaned
                continue

            overlap = FasterWhisperTranscriber._max_prefix_suffix_overlap(merged, cleaned)
            if overlap >= len(cleaned):
                continue

            if overlap <= 0:
                spacer = "" if merged.endswith(("。", "！", "？", "，", ",", ".", " ")) else " "
                merged = f"{merged}{spacer}{cleaned}".strip()
            else:
                merged = f"{merged}{cleaned[overlap:]}".strip()

        return merged

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0

    @staticmethod
    def _pcm16_to_mono_float(
        pcm16: bytes,
        channels: int,
        channel_mode: str = "mono",
    ) -> np.ndarray:
        if not pcm16:
            return np.zeros((0,), dtype=np.float32)

        usable_bytes = len(pcm16) - (len(pcm16) % 2)
        if usable_bytes <= 0:
            return np.zeros((0,), dtype=np.float32)
        if usable_bytes != len(pcm16):
            pcm16 = pcm16[:usable_bytes]

        audio = np.frombuffer(pcm16, dtype=np.int16)
        if audio.size == 0:
            return np.zeros((0,), dtype=np.float32)

        if channels > 1:
            usable = (audio.size // channels) * channels
            if usable <= 0:
                return np.zeros((0,), dtype=np.float32)

            matrix = audio[:usable].reshape(-1, channels).astype(np.float32)
            mode = channel_mode.lower().strip()
            if mode == "left":
                audio = matrix[:, 0]
            elif mode == "right":
                audio = matrix[:, min(1, channels - 1)]
            else:
                audio = matrix.mean(axis=1)
        else:
            audio = audio.astype(np.float32)

        return (audio / 32768.0).astype(np.float32)

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio.size == 0:
            return audio

        target_size = int(audio.size * dst_rate / src_rate)
        if target_size <= 1:
            return np.zeros((0,), dtype=np.float32)

        src_idx = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
        dst_idx = np.linspace(0.0, audio.size - 1, num=target_size, dtype=np.float32)
        return np.interp(dst_idx, src_idx, audio).astype(np.float32)

    @staticmethod
    def _normalize_language_hint(language: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if language is None:
            return None, None

        lang = language.strip().lower()
        if not lang or lang == "auto":
            return None, None

        if lang in {"zh-hant", "zh-tw", "zh-hk"}:
            return "zh", "hant"
        if lang in {"zh-hans", "zh-cn", "zh-sg"}:
            return "zh", "hans"

        return lang, None

    @classmethod
    def _normalize_chinese_script(cls, text: str, script: Optional[str]) -> str:
        if not text or script not in {"hant", "hans"}:
            return text

        if OpenCC is None:
            return text

        try:
            if script == "hant":
                if cls._opencc_hant is None:
                    cls._opencc_hant = OpenCC("s2twp")
                return cls._opencc_hant.convert(text)

            if cls._opencc_hans is None:
                cls._opencc_hans = OpenCC("t2s")
            return cls._opencc_hans.convert(text)
        except Exception:
            return text
