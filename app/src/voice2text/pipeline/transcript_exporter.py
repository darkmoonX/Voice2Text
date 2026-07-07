"""Session transcript exporter (txt/srt/json) with optional timestamp/speaker fields."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import threading
from typing import Callable


_FORMAT_SUFFIX: dict[str, str] = {
    "display": ".txt",
    "txt": ".txt",
    "srt": ".srt",
    "json": ".json",
}


def export_format_suffix(export_format: str) -> str:
    """Canonical file suffix for an export-format key (`display`/`txt`/`srt`/`json`)."""
    return _FORMAT_SUFFIX.get(str(export_format or "").strip().lower(), ".txt")


# Stable-token rule mirrors `whisperx_provider` (score gate + plausible word duration). Kept in
# sync deliberately so the exported `stable_ratio` matches the provider's `stability_ratio`.
_STABLE_MIN_SCORE = 0.60
_STABLE_MIN_DURATION = 0.02
_STABLE_MAX_DURATION = 1.2


def _is_stable_token(score: float, start: float, end: float) -> bool:
    duration = float(end) - float(start)
    return float(score) >= _STABLE_MIN_SCORE and _STABLE_MIN_DURATION <= duration <= _STABLE_MAX_DURATION


@dataclass
class TranscriptExportOptions:
    enabled: bool
    formats: list[str]
    include_timestamps: bool
    include_speaker: bool
    output_dir: str
    display_text_only: bool = False
    include_confidence: bool = True
    txt_confidence_annotations: bool = False


class TranscriptExporterSession:
    def __init__(self, options: TranscriptExportOptions, *, on_status: Callable[[str], None] | None = None) -> None:
        self._options = options
        self._on_status = on_status
        self._events: list[dict[str, object]] = []
        self._tokens: dict[tuple[int, int, str, str], dict[str, object]] = {}
        self._last_source = ""
        self._last_translated = ""
        self._session_started_at = datetime.now()
        self._finalized_paths: list[Path] | None = None
        self._lock = threading.Lock()

    def record(
        self,
        *,
        raw_text: str,
        source_text: str,
        translated_text: str,
        meta: dict[str, object] | None,
    ) -> None:
        if not self._options.enabled:
            return
        with self._lock:
            timestamp = datetime.now().isoformat(timespec="milliseconds")
            safe_meta = dict(meta or {})
            elapsed = self._to_float(safe_meta.get("elapsed_seconds"), 0.0)
            event: dict[str, object] = {
                "timestamp": timestamp,
                "elapsed_seconds": float(elapsed),
                "raw_text": str(raw_text or ""),
                "source_text": str(source_text or ""),
                "translated_text": str(translated_text or ""),
                "snapshot_final": bool(safe_meta.get("snapshot_final", False)),
                "snapshot_total_duration_seconds": self._to_float(
                    safe_meta.get("snapshot_total_duration_seconds"),
                    0.0,
                ),
                "token_count": int(len(safe_meta.get("token_timestamps") or []))
                if isinstance(safe_meta.get("token_timestamps"), list)
                else 0,
            }
            if self._options.include_confidence:
                event["stability_ratio"] = self._to_float(safe_meta.get("stability_ratio"), -1.0)
                event["stable_token_count"] = int(self._to_float(safe_meta.get("stable_token_count"), 0.0))
            self._events.append(event)
            self._last_source = str(source_text or self._last_source)
            self._last_translated = str(translated_text or self._last_translated)
            self._ingest_tokens(safe_meta)
            self._finalized_paths = None

    def finalize(self) -> list[Path]:
        if not self._options.enabled:
            return []
        with self._lock:
            if self._finalized_paths is not None:
                return list(self._finalized_paths)
            cues = self._build_cues()
            events_snapshot = list(self._events)
            last_source = str(self._last_source)
            last_translated = str(self._last_translated)
            token_count = int(len(self._tokens))
            confidence_summary = self._confidence_summary()
        if not cues and not last_source and not events_snapshot:
            return []

        output_dir = Path(self._options.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"transcript_{stamp}"
        written: list[Path] = []
        if self._options.display_text_only:
            path = output_dir / f"{base}.txt"
            path.write_text(last_source.strip() + "\n", encoding="utf-8")
            written.append(path)
            self._emit("Displayed subtitle text exported: " + str(path))
            with self._lock:
                self._finalized_paths = list(written)
            return written
        for fmt in self._options.formats:
            if fmt == "txt":
                path = output_dir / f"{base}.txt"
                path.write_text(self._render_txt(cues), encoding="utf-8")
                written.append(path)
            elif fmt == "srt":
                path = output_dir / f"{base}.srt"
                path.write_text(self._render_srt(cues), encoding="utf-8")
                written.append(path)
            elif fmt == "json":
                path = output_dir / f"{base}.json"
                payload = {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "session_started_at": self._session_started_at.isoformat(timespec="seconds"),
                    "options": {
                        "include_timestamps": bool(self._options.include_timestamps),
                        "include_speaker": bool(self._options.include_speaker),
                        "formats": list(self._options.formats),
                    },
                    "summary": {
                        "event_count": int(len(events_snapshot)),
                        "token_count": token_count,
                        "cue_count": int(len(cues)),
                        **confidence_summary,
                    },
                    "final_source_text": last_source,
                    "final_translated_text": last_translated,
                    "cues": cues,
                    "events": events_snapshot,
                }
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                written.append(path)
        if written:
            self._emit("Transcript exported: " + ", ".join((str(path) for path in written)))
        with self._lock:
            self._finalized_paths = list(written)
        return written

    def export_to(
        self,
        *,
        output_path: str,
        export_format: str,
        include_timestamps: bool | None = None,
        include_speaker: bool | None = None,
    ) -> Path:
        """Route a single manual export by format key.

        `display` (or empty) writes the overlay text exactly as shown; `txt`/`srt`/`json`
        render the timed/cue-based transcript through `export_single_file`.
        """
        fmt = str(export_format or "").strip().lower()
        if fmt in {"", "display"}:
            return self.export_display_text_file(output_path=output_path)
        return self.export_single_file(
            output_path=output_path,
            format_hint=fmt,
            include_timestamps=include_timestamps,
            include_speaker=include_speaker,
        )

    def export_single_file(
        self,
        *,
        output_path: str,
        format_hint: str,
        include_timestamps: bool | None = None,
        include_speaker: bool | None = None,
    ) -> Path:
        if not self._options.enabled:
            raise RuntimeError("Transcript exporter is disabled.")
        fmt = str(format_hint or "").strip().lower()
        if fmt not in {"txt", "srt", "json"}:
            raise ValueError(f"Unsupported export format: {fmt or 'empty'}")
        target = Path(str(output_path or "").strip())
        if not target.name:
            raise ValueError("Export path is empty.")
        if target.suffix.lower() != f".{fmt}":
            target = target.with_suffix(f".{fmt}")
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            cues = self._build_cues()
            events_snapshot = list(self._events)
            last_source = str(self._last_source)
            last_translated = str(self._last_translated)
            token_count = int(len(self._tokens))
            confidence_summary = self._confidence_summary()
        if not cues and not last_source and not events_snapshot:
            raise RuntimeError("No transcript data available yet.")

        use_timestamps = self._options.include_timestamps if include_timestamps is None else bool(include_timestamps)
        use_speaker = self._options.include_speaker if include_speaker is None else bool(include_speaker)

        if fmt == "txt":
            target.write_text(
                self._render_txt(cues, include_timestamps=use_timestamps, include_speaker=use_speaker),
                encoding="utf-8",
            )
        elif fmt == "srt":
            target.write_text(
                self._render_srt(cues, include_speaker=use_speaker),
                encoding="utf-8",
            )
        else:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "session_started_at": self._session_started_at.isoformat(timespec="seconds"),
                "options": {
                    "include_timestamps": bool(use_timestamps),
                    "include_speaker": bool(use_speaker),
                    "formats": [fmt],
                },
                "summary": {
                    "event_count": int(len(events_snapshot)),
                    "token_count": token_count,
                    "cue_count": int(len(cues)),
                    **confidence_summary,
                },
                "final_source_text": last_source,
                "final_translated_text": last_translated,
                "cues": cues,
                "events": events_snapshot,
            }
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._emit(f"Transcript exported: {target}")
        return target

    def export_display_text_file(self, *, output_path: str) -> Path:
        """Write the latest main overlay text exactly as displayed."""
        if not self._options.enabled:
            raise RuntimeError("Transcript exporter is disabled.")
        target = Path(str(output_path or "").strip())
        if not target.name:
            raise ValueError("Export path is empty.")
        if target.suffix.lower() != ".txt":
            target = target.with_suffix(".txt")
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            display_text = str(self._last_source or "").strip()
        if not display_text:
            raise RuntimeError("No displayed subtitle text available yet.")
        target.write_text(display_text + "\n", encoding="utf-8")
        self._emit(f"Displayed subtitle text exported: {target}")
        return target

    def _ingest_tokens(self, meta: dict[str, object]) -> None:
        rows = meta.get("token_timestamps")
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            word = str(row.get("word") or "").strip()
            if not word:
                continue
            start = self._to_float(row.get("absolute_start"), self._to_float(row.get("start"), -1.0))
            end = self._to_float(row.get("absolute_end"), self._to_float(row.get("end"), -1.0))
            if start < 0.0 or end <= start:
                continue
            # Three distinct label sources kept separate (round 0027):
            #  - effective `speaker` (profile-preferred) = the rendered/exported marker (unchanged).
            #  - `profile_speaker` = cross-window centroid identity.
            #  - `raw_speaker`     = the local per-window diarization label (`local_speaker`).
            speaker = self._normalize_export_speaker(
                str(row.get("profile_speaker") or row.get("speaker") or "").strip()
            )
            profile_speaker = self._normalize_export_speaker(str(row.get("profile_speaker") or "").strip())
            raw_speaker = self._normalize_export_speaker(str(row.get("local_speaker") or "").strip())
            # Key stays on the effective speaker so dedup behavior is unchanged.
            key = (int(round(start * 1000.0)), int(round(end * 1000.0)), word, speaker)
            if key not in self._tokens:
                self._tokens[key] = {
                    "start": float(start),
                    "end": float(end),
                    "word": word,
                    "speaker": speaker,
                    "profile_speaker": profile_speaker,
                    "raw_speaker": raw_speaker,
                    "score": self._to_float(row.get("score"), 0.0),
                }

    def _confidence_summary(self) -> dict[str, object]:
        """Session-level confidence/stability over all ingested tokens (call under lock)."""
        if not self._options.include_confidence:
            return {}
        tokens = list(self._tokens.values())
        if not tokens:
            return {"mean_confidence": 0.0, "stable_token_ratio": 0.0}
        scores = [self._to_float(token.get("score"), 0.0) for token in tokens]
        stable = sum(
            1
            for token in tokens
            if _is_stable_token(
                self._to_float(token.get("score"), 0.0),
                self._to_float(token.get("start"), 0.0),
                self._to_float(token.get("end"), 0.0),
            )
        )
        return {
            "mean_confidence": round(sum(scores) / float(len(scores)), 4),
            "stable_token_ratio": round(stable / float(len(tokens)), 4),
        }

    def _build_cues(self) -> list[dict[str, object]]:
        tokens = sorted(self._tokens.values(), key=lambda item: (float(item["start"]), float(item["end"])))
        tokens = self._smooth_micro_speaker_tokens(tokens)
        if not tokens:
            event_cues = self._build_cues_from_events()
            if event_cues:
                return event_cues
            if self._last_source.strip():
                return [{"start": 0.0, "end": 0.0, "speaker": "", "text": self._last_source.strip()}]
            return []
        cues: list[dict[str, object]] = []
        bucket: list[dict[str, object]] = []
        for token in tokens:
            if not bucket:
                bucket = [token]
                continue
            prev = bucket[-1]
            gap = float(token["start"]) - float(prev["end"])
            duration = float(token["end"]) - float(bucket[0]["start"])
            speaker_changed = str(token.get("speaker") or "") != str(prev.get("speaker") or "")
            current_text = self._join_words([str(item.get("word") or "") for item in bucket])
            hard_gap = gap > 2.0
            if hard_gap or speaker_changed:
                cues.append(self._bucket_to_cue(bucket))
                bucket = [token]
                continue
            bucket.append(token)
        if bucket:
            cues.append(self._bucket_to_cue(bucket))
        return cues

    @staticmethod
    def _smooth_micro_speaker_tokens(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
        if not tokens:
            return []
        labels = [str(item.get("speaker") or "").strip() for item in tokens]
        runs: list[dict[str, object]] = []
        start = 0
        current = labels[0]
        for idx in range(1, len(labels)):
            if labels[idx] == current:
                continue
            runs.append({"label": current, "start": start, "end": idx})
            start = idx
            current = labels[idx]
        runs.append({"label": current, "start": start, "end": len(labels)})

        out = [dict(item) for item in tokens]
        max_duration = 0.80
        max_chars = 4
        for run in runs:
            start_idx = int(run.get("start") or 0)
            end_idx = int(run.get("end") or start_idx)
            run_tokens = out[start_idx:end_idx]
            if not run_tokens:
                run["duration"] = 0.0
                run["chars"] = 0
                continue
            try:
                run["duration"] = max(
                    0.0,
                    float(run_tokens[-1].get("end") or 0.0) - float(run_tokens[0].get("start") or 0.0),
                )
            except Exception:
                run["duration"] = 0.0
            run["chars"] = sum(len(str(item.get("word") or "").strip()) for item in run_tokens)
            run["is_micro"] = (
                float(run["duration"]) <= max_duration
                or int(run["chars"]) <= max_chars
            )

        for run_idx, run in enumerate(runs):
            label = str(run.get("label") or "")
            start_idx = int(run.get("start") or 0)
            end_idx = int(run.get("end") or start_idx)
            if not label or end_idx <= start_idx:
                continue
            duration = float(run.get("duration", 0.0) or 0.0)
            char_count = int(run.get("chars", 0) or 0)
            if not bool(run.get("is_micro", False)):
                continue
            prev_run = runs[run_idx - 1] if run_idx > 0 else {}
            next_run = runs[run_idx + 1] if run_idx + 1 < len(runs) else {}
            prev_label = str(prev_run.get("label") or "")
            next_label = str(next_run.get("label") or "")
            replacement = ""
            if prev_label and next_label and prev_label == next_label:
                replacement = prev_label
            else:
                is_tiny = char_count <= 2
                if is_tiny and (prev_label or next_label):
                    if prev_label and not next_label:
                        if int(prev_run.get("chars", 0) or 0) > 2:
                            replacement = prev_label
                    elif prev_label and next_label:
                        prev_score = (
                            int(prev_run.get("chars", 0) or 0),
                            float(prev_run.get("duration", 0.0) or 0.0),
                        )
                        next_score = (
                            int(next_run.get("chars", 0) or 0),
                            float(next_run.get("duration", 0.0) or 0.0),
                        )
                        replacement = prev_label if prev_score >= next_score else next_label
            if not replacement or replacement == label:
                continue
            for idx in range(start_idx, end_idx):
                out[idx]["speaker"] = replacement
        return out

    def _build_cues_from_events(self) -> list[dict[str, object]]:
        cues: list[dict[str, object]] = []
        prev_source = ""
        for event in self._events:
            source = str(event.get("source_text") or "").strip()
            if not source:
                continue
            start = self._to_float(event.get("elapsed_seconds"), 0.0)
            is_snapshot = bool(event.get("snapshot_final", False))
            total_duration = self._to_float(event.get("snapshot_total_duration_seconds"), 0.0)
            delta = self._extract_increment_text(prev_source, source)
            prev_source = source
            if not delta:
                continue
            lines = [line.strip() for line in delta.splitlines() if line.strip()]
            if is_snapshot:
                lines = self._collapse_snapshot_lines_by_speaker(lines)
            line_duration = 0.0
            if is_snapshot and total_duration > 0.0 and lines:
                line_duration = max(0.4, total_duration / float(len(lines)))
            for idx, line in enumerate(lines):
                text = line.strip()
                if not text:
                    continue
                cue_start = float(start + (idx * line_duration)) if line_duration > 0.0 else float(start)
                speaker, body = self._parse_speaker_prefixed_line(text)
                if not body:
                    continue
                duration = line_duration if line_duration > 0.0 else max(0.4, min(3.5, 0.08 * float(len(body))))
                cues.append(
                    {
                        "start": float(cue_start),
                        "end": float(cue_start + duration),
                        "speaker": speaker,
                        "text": body,
                    }
                )
        return cues

    @classmethod
    def _collapse_snapshot_lines_by_speaker(cls, lines: list[str]) -> list[str]:
        """Final snapshots may contain visual wraps; only speaker changes should force cues."""
        out: list[str] = []
        current_speaker = ""
        current_body = ""
        for raw in lines:
            speaker, body = cls._parse_speaker_prefixed_line(str(raw or "").strip())
            if not body:
                continue
            same_speaker = bool(out) and speaker == current_speaker
            speaker_continues = bool(out) and (not speaker) and bool(current_speaker)
            no_speaker_continues = bool(out) and (not speaker) and (not current_speaker)
            if same_speaker or speaker_continues or no_speaker_continues:
                current_body = cls._join_snapshot_text(current_body, body)
                prefix = f"[{current_speaker}] " if current_speaker else ""
                out[-1] = f"{prefix}{current_body}".strip()
                continue
            current_speaker = speaker
            current_body = body
            prefix = f"[{current_speaker}] " if current_speaker else ""
            out.append(f"{prefix}{current_body}".strip())
        return out

    @classmethod
    def _parse_speaker_prefixed_line(cls, text: str) -> tuple[str, str]:
        body = str(text or "").strip()
        speaker = ""
        m = re.match(r"^\[(spk_\d+|speaker_\d+|s\d+)\]\s*(.+)$", body, flags=re.IGNORECASE)
        if m is not None:
            return (cls._normalize_export_speaker(str(m.group(1))), str(m.group(2)).strip())
        m = re.match(r"^(S\d+|SPK_\d+|SPEAKER_\d+):\s*(.+)$", body, flags=re.IGNORECASE)
        if m is not None:
            return (cls._normalize_export_speaker(str(m.group(1))), str(m.group(2)).strip())
        return (speaker, body)

    @staticmethod
    def _join_snapshot_text(left: str, right: str) -> str:
        first = str(left or "").strip()
        second = str(right or "").strip()
        if not first:
            return second
        if not second:
            return first
        if re.fullmatch(r"[\.,!?;:，。！？；：、)\]\}】》」』]", second[:1]):
            return first + second
        if re.search(r"[\u3400-\u9FFF]", first[-1:]) or re.search(r"[\u3400-\u9FFF]", second[:1]):
            return f"{first} {second}"
        return f"{first} {second}"

    @staticmethod
    def _normalize_export_speaker(label: str) -> str:
        src = str(label or "").strip()
        m = re.search(r"(\d+)", src)
        if not m:
            return src.upper()
        return f"spk_{int(m.group(1)):03d}"

    @staticmethod
    def _extract_increment_text(previous: str, current: str) -> str:
        prev = str(previous or "").strip()
        cur = str(current or "").strip()
        if not cur or cur == prev:
            return ""
        if not prev:
            return cur
        if cur.startswith(prev):
            return cur[len(prev) :].strip()
        overlap = TranscriptExporterSession._max_prefix_suffix_overlap(prev, cur)
        if overlap > 0 and overlap < len(cur):
            return cur[overlap:].strip()
        return cur

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0

    def _bucket_to_cue(self, bucket: list[dict[str, object]]) -> dict[str, object]:
        speakers: dict[str, int] = {}
        words: list[str] = []
        for token in bucket:
            spk = str(token.get("speaker") or "").strip()
            if spk:
                speakers[spk] = speakers.get(spk, 0) + 1
            words.append(str(token.get("word") or "").strip())
        speaker = max(speakers.items(), key=lambda item: item[1])[0] if speakers else ""
        cue: dict[str, object] = {
            "start": float(bucket[0]["start"]),
            "end": float(bucket[-1]["end"]),
            "speaker": speaker,
            "text": self._join_words(words),
        }
        # Round 0027: surface the three label sources separately (json only; additive). `speaker`
        # remains the effective/rendered marker used by SRT/TXT; these are observability fields.
        if self._options.include_speaker:
            cue["visible_speaker"] = speaker
            cue["profile_speaker"] = self._dominant_label(bucket, "profile_speaker")
            cue["raw_speaker"] = self._dominant_label(bucket, "raw_speaker")
        if self._options.include_confidence:
            cue.update(self._cue_confidence(bucket))
        return cue

    @staticmethod
    def _dominant_label(bucket: list[dict[str, object]], field: str) -> str:
        """Most common non-empty value of `field` across the bucket's tokens ('' if none)."""
        counts: dict[str, int] = {}
        for token in bucket:
            value = str(token.get(field) or "").strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _cue_confidence(bucket: list[dict[str, object]]) -> dict[str, object]:
        """Aggregate token `score`s into per-cue confidence fields (json only)."""
        scores: list[float] = []
        stable = 0
        for token in bucket:
            score = TranscriptExporterSession._to_float(token.get("score"), 0.0)
            scores.append(score)
            if _is_stable_token(
                score,
                TranscriptExporterSession._to_float(token.get("start"), 0.0),
                TranscriptExporterSession._to_float(token.get("end"), 0.0),
            ):
                stable += 1
        if not scores:
            return {}
        return {
            "confidence": round(sum(scores) / float(len(scores)), 4),
            "min_score": round(min(scores), 4),
            "stable_ratio": round(stable / float(len(scores)), 4),
        }

    def _render_txt(
        self,
        cues: list[dict[str, object]],
        *,
        include_timestamps: bool | None = None,
        include_speaker: bool | None = None,
    ) -> str:
        use_timestamps = self._options.include_timestamps if include_timestamps is None else bool(include_timestamps)
        use_speaker = self._options.include_speaker if include_speaker is None else bool(include_speaker)
        if not cues:
            return (self._last_source or "").strip() + "\n"
        lines: list[str] = []
        for cue in cues:
            text = str(cue.get("text") or "").strip()
            if not text:
                continue
            line = text
            if use_speaker:
                speaker = str(cue.get("speaker") or "").strip()
                if speaker:
                    line = f"[{speaker}] {line}"
            if use_timestamps:
                line = f"[{self._fmt_time_txt(float(cue.get('start') or 0.0))} -> {self._fmt_time_txt(float(cue.get('end') or 0.0))}] {line}"
            if self._options.txt_confidence_annotations and "confidence" in cue:
                line = f"{line} (conf={float(cue['confidence']):.2f})"
            lines.append(line)
        return "\n".join(lines).strip() + "\n"

    def _render_srt(self, cues: list[dict[str, object]], *, include_speaker: bool | None = None) -> str:
        use_speaker = self._options.include_speaker if include_speaker is None else bool(include_speaker)
        if not cues:
            return ""
        rows: list[str] = []
        for idx, cue in enumerate(cues, start=1):
            text = str(cue.get("text") or "").strip()
            if not text:
                continue
            speaker = str(cue.get("speaker") or "").strip()
            if use_speaker and speaker:
                text = f"{speaker}: {text}"
            start = float(cue.get("start") or 0.0)
            end = float(cue.get("end") or start)
            if end <= start:
                end = start + 0.4
            rows.append(str(idx))
            rows.append(f"{self._fmt_time_srt(start)} --> {self._fmt_time_srt(end)}")
            rows.append(text)
            rows.append("")
        return "\n".join(rows).strip() + "\n"

    @staticmethod
    def _join_words(words: list[str]) -> str:
        out: list[str] = []
        prev = ""
        for raw in words:
            token = str(raw or "").strip()
            if not token:
                continue
            if not out:
                out.append(token)
                prev = token
                continue
            if re.fullmatch(r"[\.,!?;:，。！？；：、)\]\}】》」』]", token):
                out[-1] = out[-1] + token
                prev = token
                continue
            if TranscriptExporterSession._should_join_ascii_fragment(prev, token):
                out[-1] = out[-1] + token
                prev = token
                continue
            if prev == "." and re.fullmatch(r"[A-Za-z0-9]+", token) and re.search(r"\d\.$", out[-1]):
                out[-1] = out[-1] + token
                prev = token
                continue
            if re.search(r"[\u3400-\u9FFF]", token) or re.search(r"[\u3400-\u9FFF]", prev):
                out.append(token)
            else:
                out.append(" " + token)
            prev = token
        return "".join(out).strip()

    @staticmethod
    def _contains_cjk_or_japanese(text: str) -> bool:
        return bool(re.search(r"[\u3400-\u9FFF\u3040-\u30FF]", str(text or "")))

    @staticmethod
    def _ends_with_sentence_punctuation(text: str) -> bool:
        return bool(re.search(r"[。！？!?\.]$", str(text or "").strip()))

    @staticmethod
    def _should_join_ascii_fragment(prev: str, token: str) -> bool:
        left = str(prev or "")
        right = str(token or "")
        if not left or not right:
            return False
        if not re.fullmatch(r"[A-Za-z0-9]+", left) or not re.fullmatch(r"[A-Za-z0-9]+", right):
            return False
        return len(left) == 1 or len(right) == 1

    @staticmethod
    def _fmt_time_srt(seconds: float) -> str:
        ms = max(0, int(round(seconds * 1000.0)))
        h = ms // 3600000
        ms -= h * 3600000
        m = ms // 60000
        ms -= m * 60000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _fmt_time_txt(seconds: float) -> str:
        ms = max(0, int(round(seconds * 1000.0)))
        h = ms // 3600000
        ms -= h * 3600000
        m = ms // 60000
        ms -= m * 60000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    @staticmethod
    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _emit(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)
