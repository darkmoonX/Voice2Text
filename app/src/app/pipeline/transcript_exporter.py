"""Session transcript exporter (txt/srt/json) with optional timestamp/speaker fields."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import threading
from typing import Callable


@dataclass
class TranscriptExportOptions:
    enabled: bool
    formats: list[str]
    include_timestamps: bool
    include_speaker: bool
    output_dir: str


class TranscriptExporterSession:
    def __init__(self, options: TranscriptExportOptions, *, on_status: Callable[[str], None] | None = None) -> None:
        self._options = options
        self._on_status = on_status
        self._events: list[dict[str, object]] = []
        self._tokens: dict[tuple[int, int, str, str], dict[str, object]] = {}
        self._last_source = ""
        self._last_translated = ""
        self._session_started_at = datetime.now()
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
            self._events.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": float(elapsed),
                    "raw_text": str(raw_text or ""),
                    "source_text": str(source_text or ""),
                    "translated_text": str(translated_text or ""),
                    "token_count": int(len(safe_meta.get("token_timestamps") or []))
                    if isinstance(safe_meta.get("token_timestamps"), list)
                    else 0,
                }
            )
            self._last_source = str(source_text or self._last_source)
            self._last_translated = str(translated_text or self._last_translated)
            self._ingest_tokens(safe_meta)

    def finalize(self) -> list[Path]:
        if not self._options.enabled:
            return []
        with self._lock:
            cues = self._build_cues()
            events_snapshot = list(self._events)
            last_source = str(self._last_source)
            last_translated = str(self._last_translated)
            token_count = int(len(self._tokens))
        if not cues and not last_source and not events_snapshot:
            return []

        output_dir = Path(self._options.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"transcript_{stamp}"
        written: list[Path] = []
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
        return written

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
                },
                "final_source_text": last_source,
                "final_translated_text": last_translated,
                "cues": cues,
                "events": events_snapshot,
            }
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._emit(f"Transcript exported: {target}")
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
            speaker = str(row.get("speaker") or "").strip()
            key = (int(round(start * 1000.0)), int(round(end * 1000.0)), word, speaker)
            if key not in self._tokens:
                self._tokens[key] = {
                    "start": float(start),
                    "end": float(end),
                    "word": word,
                    "speaker": speaker,
                    "score": self._to_float(row.get("score"), 0.0),
                }

    def _build_cues(self) -> list[dict[str, object]]:
        tokens = sorted(self._tokens.values(), key=lambda item: (float(item["start"]), float(item["end"])))
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
            next_word = str(token.get("word") or "")
            is_cjk_context = self._contains_cjk_or_japanese(current_text) or self._contains_cjk_or_japanese(next_word)
            hard_gap = gap > 0.2
            punctuation_boundary = self._ends_with_sentence_punctuation(current_text) and (
                duration >= 1.2 or len(current_text) >= 18
            )
            if is_cjk_context:
                too_long = duration > 10 or len(current_text) >= 60
            else:
                too_long = duration > 5.5 or len(current_text) >= 72
            if hard_gap or speaker_changed or punctuation_boundary or too_long:
                cues.append(self._bucket_to_cue(bucket))
                bucket = [token]
                continue
            bucket.append(token)
        if bucket:
            cues.append(self._bucket_to_cue(bucket))
        return cues

    def _build_cues_from_events(self) -> list[dict[str, object]]:
        cues: list[dict[str, object]] = []
        prev_source = ""
        for event in self._events:
            source = str(event.get("source_text") or "").strip()
            if not source:
                continue
            start = self._to_float(event.get("elapsed_seconds"), 0.0)
            delta = self._extract_increment_text(prev_source, source)
            prev_source = source
            if not delta:
                continue
            for line in delta.splitlines():
                text = line.strip()
                if not text:
                    continue
                speaker = ""
                body = text
                m = re.match(r"^(S\d+):\s*(.+)$", text)
                if m is not None:
                    speaker = str(m.group(1))
                    body = str(m.group(2)).strip()
                if not body:
                    continue
                duration = max(0.4, min(3.5, 0.08 * float(len(body))))
                cues.append(
                    {
                        "start": float(start),
                        "end": float(start + duration),
                        "speaker": speaker,
                        "text": body,
                    }
                )
        return cues

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
        return {
            "start": float(bucket[0]["start"]),
            "end": float(bucket[-1]["end"]),
            "speaker": speaker,
            "text": self._join_words(words),
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


