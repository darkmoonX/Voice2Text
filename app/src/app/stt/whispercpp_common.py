"""Shared helpers for whisper.cpp backends."""
from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Optional

from .audio_utils import normalize_language_hint


def normalize_whispercpp_language(language: Optional[str]) -> str:
    normalized, _script = normalize_language_hint(language)
    return normalized or "auto"


def write_mono_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16)


def read_json_object(path: Path, *, label: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{label} did not produce JSON output: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label} produced invalid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON output has unexpected shape: {path}")
    return payload


def parse_cli_segments(payload: dict) -> list[dict[str, object]]:
    raw_segments = payload.get("transcription")
    if not isinstance(raw_segments, list):
        raise RuntimeError("whisper.cpp JSON is missing transcription[]")
    segments: list[dict[str, object]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        offsets = item.get("offsets") if isinstance(item.get("offsets"), dict) else item
        start_ms = _offset_ms(offsets, "from", "start")
        end_ms = _offset_ms(offsets, "to", "end")
        if text and end_ms > start_ms:
            segments.append({"text": text, "start": start_ms / 1000.0, "end": end_ms / 1000.0})
    return segments


def parse_server_segments(payload: dict) -> list[dict[str, object]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError("whisper.cpp verbose_json is missing segments[]")
    segments: list[dict[str, object]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        start = _float_value(item.get("start"))
        end = _float_value(item.get("end"))
        if text and end > start:
            segment: dict[str, object] = {"text": text, "start": start, "end": end}
            for key in ("no_speech_prob", "avg_logprob"):
                if key in item:
                    segment[key] = _float_value(item.get(key))
            words = _parse_server_words(item.get("words"))
            if words:
                segment["words"] = words
            segments.append(segment)
    return segments


def segment_word_timestamps(segment: dict[str, object]) -> list[dict[str, object]]:
    words = segment.get("words")
    if isinstance(words, list):
        rows = [_normalize_word_timestamp(item) for item in words]
        filtered = [row for row in rows if row is not None]
        if filtered:
            return filtered
    return synthesize_segment_word_timestamps(segment)


def synthesize_segment_word_timestamps(segment: dict[str, object]) -> list[dict[str, object]]:
    text = str(segment.get("text") or "").strip()
    if not text:
        return []
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except (TypeError, ValueError):
        return []
    if not end > start:
        return []
    if re.search(r"[㐀-鿿぀-ヿ가-힯]", text):
        units = [ch for ch in text if not ch.isspace()]
    else:
        units = text.split()
    if not units:
        return []
    span = (end - start) / float(len(units))
    rows: list[dict[str, object]] = []
    for index, unit in enumerate(units):
        word_start = start + index * span
        rows.append(
            {
                "word": unit,
                "start": float(word_start),
                "end": float(word_start + span),
                "score": 1.0,
                "speaker": "",
                "profile_speaker": "",
                "local_speaker": "",
            }
        )
    return rows


def build_transcription_meta(
    *,
    provider_timing: dict[str, object],
    segments: list[dict[str, object]],
    detected_language: str = "",
    language_probabilities: object = None,
) -> dict[str, object]:
    token_meta: list[dict[str, object]] = []
    for segment in segments:
        token_meta.extend(segment_word_timestamps(segment))
    provider_timing["segment_count"] = int(len(segments))
    provider_timing["token_count"] = int(len(token_meta))
    provider_timing["stable_token_count"] = int(len(token_meta))
    meta: dict[str, object] = {
        "provider": "whispercpp",
        "stability_ratio": 1.0 if token_meta else 0.0,
        "token_count": int(len(token_meta)),
        "stable_token_count": int(len(token_meta)),
        "alignment_enabled": False,
        "token_timestamps": token_meta,
        "detected_language": detected_language,
        "speaker_turns": [],
        "speaker_turn_count": 0,
        "provider_timing": provider_timing,
    }
    if language_probabilities is not None:
        meta["language_probabilities"] = language_probabilities
    return meta


def join_segment_text(segments: list[dict[str, object]]) -> str:
    return " ".join((str(segment.get("text") or "").strip() for segment in segments if segment.get("text"))).strip()


def _offset_ms(offsets: object, primary: str, fallback: str) -> float:
    if not isinstance(offsets, dict):
        return 0.0
    return _float_value(offsets.get(primary, offsets.get(fallback, 0)))


def _float_value(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_server_words(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_word_timestamp(item)
        if normalized is not None:
            rows.append(normalized)
    return rows


def _normalize_word_timestamp(item: dict[str, object]) -> dict[str, object] | None:
    word = str(item.get("word") or "").strip()
    start = _float_value(item.get("start"))
    end = _float_value(item.get("end"))
    if not word or not end > start:
        return None
    score = _float_value(item.get("probability", item.get("score", 1.0)))
    return {
        "word": word,
        "start": start,
        "end": end,
        "score": score,
        "speaker": "",
        "profile_speaker": "",
        "local_speaker": "",
    }
