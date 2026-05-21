"""Shared audio utility helpers used by STT providers, VAD, and preprocessing pipelines."""
from __future__ import annotations
from typing import Optional
import numpy as np
from ..audio_capture import AudioChunk
try:
    from opencc import OpenCC
except Exception:
    OpenCC = None
_opencc_hant = None
_opencc_hans = None

def has_enough_signal(chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
    audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return rms >= threshold

def pcm16_to_mono_float(pcm16: bytes, channels: int, channel_mode: str='mono') -> np.ndarray:
    if not pcm16:
        return np.zeros((0,), dtype=np.float32)
    usable_bytes = len(pcm16) - len(pcm16) % 2
    if usable_bytes <= 0:
        return np.zeros((0,), dtype=np.float32)
    if usable_bytes != len(pcm16):
        pcm16 = pcm16[:usable_bytes]
    audio = np.frombuffer(pcm16, dtype=np.int16)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if channels > 1:
        usable = audio.size // channels * channels
        if usable <= 0:
            return np.zeros((0,), dtype=np.float32)
        matrix = audio[:usable].reshape(-1, channels).astype(np.float32)
        mode = channel_mode.lower().strip()
        if mode == 'left':
            audio = matrix[:, 0]
        elif mode == 'right':
            audio = matrix[:, min(1, channels - 1)]
        else:
            audio = matrix.mean(axis=1)
    else:
        audio = audio.astype(np.float32)
    return (audio / 32768.0).astype(np.float32)

def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    target_size = int(audio.size * dst_rate / src_rate)
    if target_size <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, audio.size - 1, num=target_size, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)

def to_int16_pcm(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()

def normalize_language_hint(language: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if language is None:
        return (None, None)
    lang = language.strip().lower()
    if not lang or lang == 'auto':
        return (None, None)
    if lang in {'zh-hant', 'zh-tw', 'zh-hk'}:
        return ('zh', 'hant')
    if lang in {'zh-hans', 'zh-cn', 'zh-sg'}:
        return ('zh', 'hans')
    return (lang, None)

def normalize_chinese_script(text: str, script: Optional[str]) -> str:
    if not text or script not in {'hant', 'hans'}:
        return text
    if OpenCC is None:
        return text
    global _opencc_hant, _opencc_hans
    try:
        if script == 'hant':
            if _opencc_hant is None:
                _opencc_hant = OpenCC('s2twp')
            return _opencc_hant.convert(text)
        if _opencc_hans is None:
            _opencc_hans = OpenCC('t2s')
        return _opencc_hans.convert(text)
    except Exception:
        return text
