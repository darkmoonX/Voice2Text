"""Audio mixing and resampling utilities for Python capture backends."""
from __future__ import annotations

import numpy as np


def pcm16_to_mono_float(pcm16: bytes, channels: int, channel_mode: str = "mono") -> np.ndarray:
    if not pcm16:
        return np.zeros((0,), dtype=np.float32)
    audio = np.frombuffer(pcm16, dtype=np.int16)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if channels > 1:
        usable = audio.size // channels * channels
        if usable <= 0:
            return np.zeros((0,), dtype=np.float32)
        matrix = audio[:usable].reshape(-1, channels).astype(np.float32)
        mode = channel_mode.lower().strip()
        if mode == "left":
            mixed = matrix[:, 0]
        elif mode == "right":
            mixed = matrix[:, min(1, channels - 1)]
        else:
            mixed = matrix.mean(axis=1)
    else:
        mixed = audio.astype(np.float32)
    return (mixed / 32768.0).astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    target_size = int(audio.size * dst_rate / max(1, src_rate))
    if target_size <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, audio.size - 1, num=target_size, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


def mix_tracks(tracks: list[tuple[float, np.ndarray]]) -> np.ndarray:
    max_len = max((track.size for (_, track) in tracks))
    mix = np.zeros((max_len,), dtype=np.float32)
    total_weight = 0.0
    for (weight, track) in tracks:
        if track.size < max_len:
            padded = np.pad(track, (0, max_len - track.size), mode="constant")
        else:
            padded = track
        mix += padded * float(weight)
        total_weight += abs(float(weight))
    if total_weight <= 0.0:
        total_weight = float(len(tracks))
    return np.clip(mix / total_weight, -1.0, 1.0)
