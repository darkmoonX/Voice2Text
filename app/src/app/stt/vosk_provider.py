"""Vosk offline STT provider adapter with preset-model auto-download support."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Optional
from ..audio_capture import AudioChunk
from ..model_paths import library_model_dir
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample, to_int16_pcm
from .model_assets import ensure_model_preset_downloaded

class VoskTranscriber:
    """Vosk provider wrapper used by STT factory."""

    def __init__(self, model_ref: str='small', target_sample_rate: int=16000, auto_download: bool=True, progress_callback: Callable[[str], None] | None=None) -> None:
        try:
            from vosk import KaldiRecognizer, Model
        except Exception as exc:
            raise RuntimeError('vosk is not installed. Run: pip install vosk') from exc
        self._target_sample_rate = int(target_sample_rate)
        self._recognizer_cls = KaldiRecognizer
        model_path = self._resolve_model_path(model_ref, auto_download=auto_download, progress_callback=progress_callback)
        self._model = Model(str(model_path))

    def has_enough_signal(self, chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk: AudioChunk, language: Optional[str]=None, channel_mode: str='mono') -> str:
        """Decode a chunk using Vosk recognizer (language hint only affects script postprocess)."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return ''
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 4:
            return ''
        (_, zh_script) = normalize_language_hint(language)
        recognizer = self._recognizer_cls(self._model, float(self._target_sample_rate))
        try:
            recognizer.SetWords(False)
        except Exception:
            pass
        pcm = to_int16_pcm(audio)
        try:
            recognizer.AcceptWaveform(pcm)
            payload = recognizer.FinalResult()
        except Exception:
            payload = recognizer.Result()
        text = self._extract_text(payload)
        return normalize_chinese_script(text, zh_script)

    def _resolve_model_path(self, model_ref: str, auto_download: bool, progress_callback: Callable[[str], None] | None) -> Path:
        model_ref = model_ref.strip() or 'small'
        model_path = Path(model_ref)
        if model_path.exists():
            return model_path
        if '/' in model_ref or '\\' in model_ref:
            raise FileNotFoundError(f'Vosk model path not found: {model_ref}')
        model_root = library_model_dir('vosk')
        candidate = model_root / model_ref
        if candidate.exists():
            return candidate
        if auto_download:
            downloaded = ensure_model_preset_downloaded(provider='vosk', model_ref=model_ref, model_root=model_root, progress_callback=progress_callback)
            if downloaded is not None and downloaded.exists():
                return downloaded
        dirs = sorted([item for item in model_root.iterdir() if item.is_dir()])
        if len(dirs) == 1:
            return dirs[0]
        raise FileNotFoundError(f'Unable to locate Vosk model. Set --stt-model-path to a Vosk model directory. Searched: {candidate}')

    @staticmethod
    def _extract_text(payload: object) -> str:
        if payload is None:
            return ''
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return payload.strip()
        elif isinstance(payload, dict):
            data = payload
        else:
            return str(payload).strip()
        text = str(data.get('text', '') or '').strip()
        if text:
            return text
        alts = data.get('alternatives') or []
        if isinstance(alts, list):
            for item in alts:
                if isinstance(item, dict):
                    candidate = str(item.get('text', '') or '').strip()
                    if candidate:
                        return candidate
        return ''
