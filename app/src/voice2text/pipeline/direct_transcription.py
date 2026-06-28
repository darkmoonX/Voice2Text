"""Shared whole-file imported-audio transcription helpers."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import wave
from typing import Callable

from ..capture import AudioChunk
from ..config import RuntimeConfig
from ..stt.registry import normalize_stt_provider

ProgressCallback = Callable[[float, float], None]
StatusCallback = Callable[[str], None]


def resolve_ffmpeg(ffmpeg_dir: str = "") -> str:
    """Resolve an ffmpeg executable the same way the rest of the product does.

    Honors the configured ``ffmpeg_dll_dir`` (matching ``audio_capture._resolve_ffmpeg``)
    before falling back to the system PATH, so direct import obeys the user's ffmpeg
    setting instead of a baked-in path.
    """
    directory = str(ffmpeg_dir or "").strip()
    if directory:
        for name in ("ffmpeg.exe", "ffmpeg"):
            candidate = Path(directory) / name
            if candidate.exists():
                return str(candidate)
    return str(shutil.which("ffmpeg") or "")


def decode_to_wav_16k_mono(input_path: Path, *, ffmpeg_dir: str = "") -> Path:
    suffix = input_path.suffix.lower()
    if suffix == ".wav":
        return input_path
    ffmpeg = resolve_ffmpeg(ffmpeg_dir)
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg executable was not found; direct import requires FFmpeg for non-WAV input."
        )
    tmp = Path(tempfile.mkdtemp(prefix="v2t_direct_")) / "decoded.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ffmpeg decode failed for {input_path}: {exc}") from exc
    if proc.returncode == 0 and tmp.exists():
        return tmp
    err = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
    raise RuntimeError(f"ffmpeg decode failed for {input_path}: {err}")


def read_wav(path: Path) -> AudioChunk:
    with wave.open(str(path), "rb") as wf:
        sample_rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        pcm16 = wf.readframes(wf.getnframes())
    return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=channels)


def audio_duration_seconds(chunk: AudioChunk) -> float:
    bytes_per_second = float(max(1, int(chunk.sample_rate) * int(chunk.channels) * 2))
    return float(len(chunk.pcm16)) / bytes_per_second


def slice_audio_chunk(chunk: AudioChunk, start_seconds: float, duration_seconds: float) -> AudioChunk:
    frame_size = max(1, int(chunk.channels) * 2)
    start_frame = max(0, int(round(float(start_seconds) * float(chunk.sample_rate))))
    frame_count = max(0, int(round(float(duration_seconds) * float(chunk.sample_rate))))
    start_byte = min(len(chunk.pcm16), start_frame * frame_size)
    end_byte = min(len(chunk.pcm16), start_byte + frame_count * frame_size)
    return AudioChunk(
        pcm16=chunk.pcm16[start_byte:end_byte],
        sample_rate=chunk.sample_rate,
        channels=chunk.channels,
    )


def resolve_direct_chunk_seconds(requested_chunk_seconds: float, duration_seconds: float) -> tuple[float, str]:
    requested = float(max(0.0, requested_chunk_seconds))
    if requested > 0.0:
        return (requested, "requested")
    return (0.0, "single")


def enrich_absolute_timestamps(meta: dict[str, object], elapsed_seconds: float) -> dict[str, object]:
    out = dict(meta or {})
    rows = out.get("token_timestamps")
    if not isinstance(rows, list):
        out["elapsed_seconds"] = float(elapsed_seconds)
        return out
    enriched: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
            item["absolute_start"] = float(elapsed_seconds + start)
            item["absolute_end"] = float(elapsed_seconds + end)
        except Exception:
            pass
        enriched.append(item)
    out["token_timestamps"] = enriched
    out["elapsed_seconds"] = float(elapsed_seconds)
    return out


def apply_speaker_profile_remap_to_meta(meta: dict[str, object], remap: dict[str, str]) -> dict[str, object]:
    if not remap:
        return dict(meta)
    out = dict(meta)

    def _mapped(value: object) -> object:
        text = str(value or "").strip()
        return remap.get(text, value)

    rows = out.get("token_timestamps")
    if isinstance(rows, list):
        mapped_rows: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            for key in ("speaker", "profile_speaker"):
                if key in item:
                    item[key] = _mapped(item.get(key))
            mapped_rows.append(item)
        out["token_timestamps"] = mapped_rows

    turns = out.get("speaker_turns")
    if isinstance(turns, list):
        mapped_turns: list[dict[str, object]] = []
        for row in turns:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            for key in ("speaker", "profile_speaker"):
                if key in item:
                    item[key] = _mapped(item.get(key))
            mapped_turns.append(item)
        out["speaker_turns"] = mapped_turns
    return out


def call_speaker_profile_reconcile(reconcile: object, *, threshold: float) -> dict[str, object]:
    if not callable(reconcile):
        return {"status": "skip_unavailable", "merged_count": 0, "remap": {}}
    try:
        if threshold > 0.0:
            stats = reconcile(threshold=float(threshold))
        else:
            stats = reconcile()
    except TypeError:
        stats = reconcile()
    if not isinstance(stats, dict):
        return {"status": "invalid_result", "merged_count": 0, "remap": {}}
    return dict(stats)


def reconcile_direct_speaker_profiles(
    transcriber: object,
    metas: list[dict[str, object]],
    *,
    threshold: float = 0.0,
) -> dict[str, object]:
    reconcile = getattr(transcriber, "reconcile_speaker_profiles", None)
    try:
        stats = call_speaker_profile_reconcile(reconcile, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc), "merged_count": 0, "remap": {}}
    raw_remap = stats.get("remap")
    remap = {str(k): str(v) for (k, v) in raw_remap.items()} if isinstance(raw_remap, dict) else {}
    if remap:
        for index, meta in enumerate(list(metas)):
            metas[index] = apply_speaker_profile_remap_to_meta(meta, remap)
    return dict(stats)


def _assign_global_speakers(
    token_timestamps: list[dict[str, object]],
    turns: list[dict[str, object]],
) -> int:
    """Stamp each token with the whole-file diarization speaker by time overlap.

    Tokens carry absolute audio times (``absolute_start``/``absolute_end`` added by
    ``enrich_absolute_timestamps``, falling back to ``start``/``end``). The turn whose
    span best overlaps a token's midpoint wins; the label is written to ``speaker``,
    ``profile_speaker`` and ``local_speaker`` so every downstream consumer (export cues,
    text markers, scorer) sees the same globally-consistent identity. Returns the number
    of tokens that received a label.
    """
    if not turns:
        return 0
    spans = [(float(t["start"]), float(t["end"]), str(t["speaker"])) for t in turns]
    assigned = 0
    for row in token_timestamps:
        try:
            start = float(row.get("absolute_start", row.get("start")))
            end = float(row.get("absolute_end", row.get("end")))
        except Exception:
            continue
        mid = (start + end) / 2.0
        best_label = ""
        best_overlap = 0.0
        for s, e, label in spans:
            if s <= mid < e:
                best_label = label
                break
            overlap = min(end, e) - max(start, s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label:
            row["speaker"] = best_label
            row["profile_speaker"] = best_label
            row["local_speaker"] = best_label
            assigned += 1
    return assigned


def run_direct_transcription(
    cfg: RuntimeConfig,
    full_audio: AudioChunk,
    *,
    transcriber: object,
    chunk_seconds: float,
    language_subchunk_seconds: float = 30.0,
    speaker_profile_reconcile_threshold: float = 0.0,
    whole_file_diarization: bool = True,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
) -> dict[str, object]:
    """Run one whole-file imported-audio transcription pass with optional chunking.

    When ``whole_file_diarization`` is set and the transcriber supports it (round 0045),
    per-chunk diarization + the cross-window profile re-cluster are suppressed during the
    ASR loop and a single whole-file diarization pass assigns globally-consistent speaker
    labels afterwards. This avoids the chunked-diarization + weaker ``pyannote/embedding``
    collapse that merged distinct speakers (e.g. zh Bn 3 voices -> 1).
    """
    provider = normalize_stt_provider(str(getattr(cfg, "stt_provider", "whisperx") or "whisperx"))
    if provider == "whispercpp" and on_status is not None:
        on_status("direct mode: whispercpp has no diarization - single-pass, no speaker labels")

    set_suppressed = getattr(transcriber, "set_diarization_suppressed", None)
    diarize_whole_file = getattr(transcriber, "diarize_whole_file_turns", None)
    supports_whole_file = bool(getattr(transcriber, "supports_whole_file_diarization", lambda: False)())
    whole_file_active = (
        bool(whole_file_diarization)
        and supports_whole_file
        and callable(set_suppressed)
        and callable(diarize_whole_file)
    )

    duration = audio_duration_seconds(full_audio)
    requested_chunk_seconds = float(max(0.0, chunk_seconds))
    resolved_chunk_seconds, chunk_mode = resolve_direct_chunk_seconds(requested_chunk_seconds, duration)
    use_chunked = resolved_chunk_seconds > 0.0 and duration > resolved_chunk_seconds
    if on_status is not None:
        mode = "chunked" if use_chunked else "single"
        suffix = f"; chunk={resolved_chunk_seconds:.2f}s" if use_chunked else ""
        on_status(f"direct mode: audio duration={duration:.2f}s; mode={mode}{suffix}")

    texts: list[str] = []
    metas: list[dict[str, object]] = []
    offset = 0.0
    index = 0
    language_subchunk_seconds = float(max(0.0, language_subchunk_seconds))
    use_language_subchunks = (
        cfg.source_language is None
        and language_subchunk_seconds > 0.0
        and resolved_chunk_seconds > language_subchunk_seconds
    )

    def _emit_progress(completed_audio_seconds: float) -> None:
        if on_progress is not None:
            on_progress(min(float(duration), max(0.0, float(completed_audio_seconds))), float(duration))

    get_meta = getattr(transcriber, "get_last_transcription_meta", lambda: {})
    transcribe = getattr(transcriber, "transcribe")
    if whole_file_active:
        # Suppress per-chunk diarization + profile re-cluster during the ASR loop; a
        # single whole-file pass below assigns globally-consistent speaker labels.
        set_suppressed(True)
        if on_status is not None:
            on_status("direct mode: whole-file diarization (per-chunk diarization suppressed)")
    try:
        while offset < duration or (index == 0 and duration == 0.0):
            index += 1
            current_duration = duration if not use_chunked else min(resolved_chunk_seconds, max(0.0, duration - offset))
            if current_duration <= 0.0:
                break
            audio = full_audio if not use_chunked else slice_audio_chunk(full_audio, offset, current_duration)
            if on_status is not None:
                on_status(f"direct mode: chunk {index} start={offset:.2f}s duration={current_duration:.2f}s")
            if use_language_subchunks and current_duration > language_subchunk_seconds:
                sub_offset = 0.0
                sub_index = 0
                while sub_offset < current_duration:
                    sub_index += 1
                    sub_duration = min(language_subchunk_seconds, max(0.0, current_duration - sub_offset))
                    if sub_duration <= 0.0:
                        break
                    sub_audio = slice_audio_chunk(audio, sub_offset, sub_duration)
                    if on_status is not None:
                        on_status(
                            f"direct mode: chunk {index}.{sub_index} start={offset + sub_offset:.2f}s "
                            f"duration={sub_duration:.2f}s language=auto"
                        )
                    text = transcribe(sub_audio, language=None, channel_mode=cfg.source_channel_mode)
                    meta = get_meta()
                    if not isinstance(meta, dict):
                        meta = {}
                    meta = enrich_absolute_timestamps(meta, offset + sub_offset)
                    meta["direct_parent_chunk_index"] = int(index)
                    meta["direct_language_subchunk_index"] = int(sub_index)
                    texts.append(str(text or "").strip())
                    metas.append(meta)
                    sub_offset += language_subchunk_seconds
                    _emit_progress(offset + min(sub_offset, current_duration))
            else:
                text = transcribe(audio, language=cfg.source_language, channel_mode=cfg.source_channel_mode)
                meta = get_meta()
                if not isinstance(meta, dict):
                    meta = {}
                meta = enrich_absolute_timestamps(meta, offset)
                texts.append(str(text or "").strip())
                metas.append(meta)
                _emit_progress(offset + current_duration)
            if not use_chunked:
                break
            offset += resolved_chunk_seconds
    finally:
        if whole_file_active:
            set_suppressed(False)

    text = "\n".join((item for item in texts if item)).strip()
    token_timestamps: list[dict[str, object]] = []
    detected_language = ""
    alignment_language = ""
    for meta in metas:
        if not detected_language:
            detected_language = str(meta.get("detected_language") or "")
        if not alignment_language:
            alignment_language = str(meta.get("alignment_language") or "")
        rows = meta.get("token_timestamps")
        if isinstance(rows, list):
            token_timestamps.extend([dict(row) for row in rows if isinstance(row, dict)])

    if whole_file_active:
        # One whole-file diarization pass -> globally-consistent labels assigned by time.
        # No cross-window profile re-cluster (its weaker pyannote/embedding is what
        # collapsed distinct speakers); reconciliation is intentionally skipped.
        if on_status is not None:
            on_status("direct mode: running whole-file diarization pass")
        turns = diarize_whole_file(full_audio, cfg.source_channel_mode)
        assigned = _assign_global_speakers(token_timestamps, turns)
        speaker_count = len({str(t.get("speaker") or "") for t in turns if str(t.get("speaker") or "")})
        if on_status is not None:
            on_status(
                f"direct mode: whole-file diarization assigned {assigned}/{len(token_timestamps)} "
                f"tokens across {speaker_count} speaker(s)"
            )
        speaker_reconciliation = {
            "status": "whole_file_diarization",
            "merged_count": 0,
            "remap": {},
            "turn_count": len(turns),
            "speaker_count": speaker_count,
            "tokens_assigned": assigned,
        }
    else:
        speaker_reconciliation = reconcile_direct_speaker_profiles(
            transcriber,
            metas,
            threshold=float(max(0.0, speaker_profile_reconcile_threshold)),
        )
        # reconcile may rewrite metas in place; re-extract so combined tokens reflect it.
        token_timestamps = []
        for meta in metas:
            rows = meta.get("token_timestamps")
            if isinstance(rows, list):
                token_timestamps.extend([dict(row) for row in rows if isinstance(row, dict)])
    combined_meta: dict[str, object] = {
        "elapsed_seconds": 0.0,
        "token_timestamps": token_timestamps,
        "token_count": len(token_timestamps),
        "stable_token_count": sum(
            1
            for row in token_timestamps
            if float(row.get("score", 0.0) or 0.0) >= 0.60
        ),
        "detected_language": detected_language,
        "alignment_language": alignment_language,
        "audio_duration_seconds": duration,
        "direct_chunk_count": len(metas),
        "direct_chunk_seconds": resolved_chunk_seconds if use_chunked else 0.0,
        "direct_chunk_mode": chunk_mode,
        "direct_auto_chunked": bool(chunk_mode == "auto" and use_chunked),
        "direct_language_subchunk_seconds": float(language_subchunk_seconds if use_language_subchunks else 0.0),
        "direct_requested_chunk_seconds": requested_chunk_seconds,
        "speaker_profile_reconciliation": speaker_reconciliation,
    }
    return {"text": text, "meta": combined_meta}
