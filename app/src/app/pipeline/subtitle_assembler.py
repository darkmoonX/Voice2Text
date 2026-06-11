"""Subtitle assembly module.

Time-aligned merge strategy:
- `history`: immutable words earlier than current raw window start.
- `stable`: agreed words (count >= agreement_count) still in the active region.
- `partial`: candidate words in the active region.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time


@dataclass
class _WordState:
    word: str
    start: float
    end: float
    score: float
    count: int
    last_seen: float
    speaker: str = ''


class SubtitleAssembler:
    _CJK_MAX_COMPACT_CHARS = 18

    def __init__(self) -> None:
        self._is_cjk_source = False
        self._auto_detect_cjk = True
        self._cjk_no_space_gap_seconds = 0.1
        self._speaker_marker_style = 'spk'
        self.reset()

    def reset(self) -> None:
        self._history_words: list[_WordState] = []
        self._stable_words: list[_WordState] = []
        self._partial_words: list[_WordState] = []
        self._latest_partial_text = ''
        self._last_emitted_source_text = ''
        self._speaker_display_map: dict[str, str] = {}
        self._speaker_display_next_index = 0
        self._required_agreement_count = 3
        self._score_threshold = 0.60
        self._speaker_switch_confirm_tokens = 2
        self._speaker_switch_min_duration_seconds = 0.18
        self._speaker_micro_turn_max_duration_seconds = 0.80
        self._speaker_micro_turn_max_chars = 4
        self._last_cjk_spacing_summary: dict[str, object] = {}
        self._last_speaker_smoothing_summary: dict[str, object] = {}
        self._last_merge_diagnostics: dict[str, object] = {}

    def set_language_context(self, source_language: str | None) -> None:
        token = (source_language or '').strip().lower()
        if token in {'', 'auto', 'none'}:
            self._is_cjk_source = False
            self._auto_detect_cjk = True
            return
        self._auto_detect_cjk = False
        self._is_cjk_source = token.startswith('zh')

    def set_cjk_no_space_gap_seconds(self, seconds: float) -> None:
        try:
            value = float(seconds)
        except Exception:
            value = 0.2
        self._cjk_no_space_gap_seconds = max(0.0, min(3.0, value))

    def set_speaker_marker_style(self, style: str | None) -> None:
        token = str(style or '').strip().lower()
        self._speaker_marker_style = 'arrow' if token in {'arrow', 'arrows', '>>'} else 'spk'

    def merge_incremental_text(self, text: str, *, overlap_merge_method: str, segment_seconds: float, hop_seconds: float, transcription_meta: dict[str, object] | None = None) -> str:
        _ = (overlap_merge_method, segment_seconds, hop_seconds)
        total_started_at = time.perf_counter()
        diagnostics: dict[str, object] = {
            'incoming_count': 0,
            'history_count_before': int(len(self._history_words)),
            'stable_count_before': int(len(self._stable_words)),
            'partial_count_before': int(len(self._partial_words)),
            'cleaned_chars': 0,
            'history_chars': 0,
            'visible_chars': 0,
            'merged_chars': 0,
            'returned_empty': False,
        }
        step_started_at = time.perf_counter()
        cleaned = self._normalize_output_text(text)
        diagnostics['normalize_seconds'] = time.perf_counter() - step_started_at
        diagnostics['cleaned_chars'] = int(len(cleaned))
        self._latest_partial_text = cleaned
        if not cleaned:
            diagnostics['returned_empty'] = True
            diagnostics['total_seconds'] = time.perf_counter() - total_started_at
            diagnostics['history_count_after'] = int(len(self._history_words))
            diagnostics['stable_count_after'] = int(len(self._stable_words))
            diagnostics['partial_count_after'] = int(len(self._partial_words))
            self._last_merge_diagnostics = diagnostics
            return ''

        meta = transcription_meta or {}
        elapsed = self._to_float(meta.get('elapsed_seconds', 0.0), 0.0)
        step_started_at = time.perf_counter()
        incoming = self._extract_incoming_words(meta, elapsed)
        diagnostics['extract_seconds'] = time.perf_counter() - step_started_at
        diagnostics['incoming_count'] = int(len(incoming))
        if incoming and self._auto_detect_cjk:
            self._is_cjk_source = self._contains_cjk_in_words(incoming)

        step_started_at = time.perf_counter()
        if incoming:
            raw_start = min((w.start for w in incoming), default=elapsed)
            self._flush_stable_to_history(raw_start)
            self._merge_incoming_words(incoming)
            self._promote_partial_to_stable()
            self._prune_active_words(raw_start)
        else:
            raw_start = elapsed
            self._flush_stable_to_history(raw_start)
        diagnostics['state_update_seconds'] = time.perf_counter() - step_started_at
        diagnostics['raw_start'] = float(raw_start)
        diagnostics['history_count_after_state'] = int(len(self._history_words))
        diagnostics['stable_count_after_state'] = int(len(self._stable_words))
        diagnostics['partial_count_after_state'] = int(len(self._partial_words))

        step_started_at = time.perf_counter()
        if incoming:
            self._latest_partial_text = self._words_to_text(self._partial_words)
        else:
            self._latest_partial_text = cleaned
        diagnostics['partial_render_seconds'] = time.perf_counter() - step_started_at

        # Output policy:
        # - keep token-level state updates (history/stable/partial) for stability bookkeeping
        # - but compose visible text as history + current raw overlap-merge to avoid
        #   duplicated tails from stable/partial rendering overlap.
        step_started_at = time.perf_counter()
        history_text = self._words_to_text(self._history_words)
        diagnostics['history_render_seconds'] = time.perf_counter() - step_started_at
        diagnostics['history_chars'] = int(len(history_text))
        step_started_at = time.perf_counter()
        visible_cleaned = self._apply_cjk_pause_spacing_to_text(cleaned, incoming)
        diagnostics['spacing_seconds'] = time.perf_counter() - step_started_at
        diagnostics['visible_chars'] = int(len(visible_cleaned))
        step_started_at = time.perf_counter()
        merged = self._merge_by_exact_overlap(history_text, visible_cleaned)
        diagnostics['overlap_seconds'] = time.perf_counter() - step_started_at
        step_started_at = time.perf_counter()
        merged = self._normalize_output_text(merged)[-1800:]
        diagnostics['final_normalize_seconds'] = time.perf_counter() - step_started_at
        diagnostics['merged_chars'] = int(len(merged))
        diagnostics['history_count_after'] = int(len(self._history_words))
        diagnostics['stable_count_after'] = int(len(self._stable_words))
        diagnostics['partial_count_after'] = int(len(self._partial_words))
        if merged == self._last_emitted_source_text:
            diagnostics['returned_empty'] = True
            diagnostics['total_seconds'] = time.perf_counter() - total_started_at
            self._last_merge_diagnostics = diagnostics
            return ''
        self._last_emitted_source_text = merged
        diagnostics['total_seconds'] = time.perf_counter() - total_started_at
        self._last_merge_diagnostics = diagnostics
        return merged

    def _extract_incoming_words(self, meta: dict[str, object], elapsed: float) -> list[_WordState]:
        items = meta.get('token_timestamps')
        if not isinstance(items, list):
            return []
        out: list[_WordState] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            word = str(raw.get('word') or '').strip()
            if not word:
                continue
            score = self._to_float(raw.get('score'), 0.0)
            if score < self._score_threshold:
                continue
            start_rel = self._to_float(raw.get('start'), -1.0)
            end_rel = self._to_float(raw.get('end'), -1.0)
            if start_rel < 0.0 or end_rel <= start_rel:
                continue
            start_abs = elapsed + start_rel
            end_abs = elapsed + end_rel
            out.append(
                _WordState(
                    word=word,
                    start=start_abs,
                    end=end_abs,
                    score=score,
                    count=1,
                    last_seen=end_abs,
                    speaker=str(raw.get('speaker') or '').strip(),
                )
            )
        out.sort(key=lambda w: (w.start, w.end))
        return out

    def _merge_incoming_words(self, incoming: list[_WordState]) -> None:
        for word in incoming:
            target = self._find_match(self._stable_words, word)
            if target is not None:
                self._update_word(target, word)
                continue
            target = self._find_match(self._partial_words, word)
            if target is not None:
                self._update_word(target, word)
            else:
                self._partial_words.append(word)

    def _promote_partial_to_stable(self) -> None:
        keep_partial: list[_WordState] = []
        for word in self._partial_words:
            if word.count >= self._required_agreement_count:
                self._stable_words.append(word)
            else:
                keep_partial.append(word)
        self._partial_words = keep_partial
        self._stable_words = self._dedupe_words(self._stable_words)

    def _flush_stable_to_history(self, raw_start: float) -> None:
        if not self._stable_words:
            return
        remain: list[_WordState] = []
        moved: list[_WordState] = []
        for word in self._stable_words:
            if word.end <= raw_start:
                moved.append(word)
            else:
                remain.append(word)
        self._stable_words = remain
        if moved:
            self._history_words.extend(moved)
            self._history_words.sort(key=lambda w: (w.start, w.end))
            self._history_words = self._dedupe_words(self._history_words)

    def _prune_active_words(self, raw_start: float) -> None:
        # Keep partial near current raw window; older unresolved words are dropped.
        cutoff = raw_start - 1.0
        self._partial_words = [w for w in self._partial_words if w.end >= cutoff]
        self._partial_words.sort(key=lambda w: (w.start, w.end))

    def mark_sentence_break(self) -> None:
        # Force current confirmed words into immutable history on sentence break.
        if self._stable_words:
            self._history_words.extend(self._stable_words)
            self._history_words.sort(key=lambda w: (w.start, w.end))
            self._history_words = self._dedupe_words(self._history_words)
            self._stable_words = []

    def get_stable_text(self) -> str:
        return self._words_to_text(self._stable_words)

    def get_partial_text(self) -> str:
        return self._latest_partial_text


    def get_history_text(self) -> str:
        return self._words_to_text(self._history_words)

    def get_history_state(self) -> list[dict[str, object]]:
        return self._words_to_state(self._history_words)

    def get_partial_state(self) -> list[dict[str, object]]:
        return self._words_to_state(self._partial_words)

    def get_stable_state(self) -> list[dict[str, object]]:
        return self._words_to_state(self._stable_words)

    def get_debug_summary(self) -> dict[str, object]:
        return {
            'history_count': len(self._history_words),
            'stable_count': len(self._stable_words),
            'partial_count': len(self._partial_words),
            'is_cjk_source': bool(self._is_cjk_source),
            'auto_detect_cjk': bool(self._auto_detect_cjk),
            'cjk_spacing': dict(self._last_cjk_spacing_summary),
            'speaker_smoothing': dict(self._last_speaker_smoothing_summary),
            'merge_diagnostics': dict(self._last_merge_diagnostics),
        }

    def get_last_merge_diagnostics(self) -> dict[str, object]:
        return dict(self._last_merge_diagnostics)

    @staticmethod
    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    @staticmethod
    def _normalize_word(word: str) -> str:
        return word.strip().lower().replace(' ', '')

    @staticmethod
    def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        return (inter / union) if union > 0 else 0.0

    def _word_matches(self, a: _WordState, b: _WordState, *, max_start_diff: float = 0.25, max_end_diff: float = 0.35, min_iou: float = 0.30) -> bool:
        if self._normalize_word(a.word) != self._normalize_word(b.word):
            return False
        if abs(a.start - b.start) > max_start_diff:
            return False
        if abs(a.end - b.end) > max_end_diff:
            return False
        if self._interval_iou(a.start, a.end, b.start, b.end) < min_iou:
            return False
        return True

    def _find_match(self, pool: list[_WordState], candidate: _WordState) -> _WordState | None:
        for item in pool:
            if self._word_matches(item, candidate):
                return item
        return None

    @staticmethod
    def _update_word(target: _WordState, incoming: _WordState) -> None:
        target_weight = max(1, int(target.count))
        incoming_weight = max(1, int(incoming.count))
        total_weight = target_weight + incoming_weight
        target.start = ((target.start * target_weight) + (incoming.start * incoming_weight)) / float(total_weight)
        target.end = ((target.end * target_weight) + (incoming.end * incoming_weight)) / float(total_weight)
        target.score = ((target.score * target_weight) + (incoming.score * incoming_weight)) / float(total_weight)
        target.count = target_weight + incoming_weight
        target.last_seen = max(target.last_seen, incoming.last_seen)
        if (not target.speaker) and incoming.speaker:
            target.speaker = incoming.speaker
        elif incoming.speaker and target.speaker and (incoming.speaker != target.speaker):
            if float(incoming.score) >= float(target.score):
                target.speaker = incoming.speaker

    def _words_to_text(self, words: list[_WordState]) -> str:
        if not words:
            return ''
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        speaker_labels = self._stabilize_speakers(ordered)
        out: list[str] = []
        prev: _WordState | None = None
        last_speaker_label = ''
        for (w, speaker) in zip(ordered, speaker_labels):
            raw_token = str(w.word or '')
            token = self._normalize_cjk_token(raw_token) if self._is_cjk_source else raw_token.strip()
            if not token:
                continue
            marker = self._speaker_label_to_marker(speaker)
            if marker and speaker and (speaker != last_speaker_label):
                if self._is_cjk_source:
                    out.append(f'\n{marker} ')
                else:
                    if marker == '>>':
                        out.append(f'\n{marker}' if out else f'{marker}')
                    else:
                        out.append(f'\n{marker}' if out else f'{marker}')
                last_speaker_label = speaker
            if not self._is_cjk_source:
                out.append(token)
                continue
            if prev is None:
                out.append(token)
                prev = w
                continue
            gap = max(0.0, float(w.start) - float(prev.end))
            needs_space = gap > self._cjk_no_space_gap_seconds
            if self._is_punct(token):
                needs_space = False
            if needs_space:
                out.append(' ')
            out.append(token)
            prev = w
        if not self._is_cjk_source:
            text = ' '.join(out)
            lines = []
            for line in text.splitlines():
                cleaned = re.sub(r'[ \t]+', ' ', line).strip()
                if cleaned:
                    lines.append(cleaned)
            return '\n'.join(lines).strip()
        text = ''.join(out)
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r'[ \t]+', ' ', line).strip()
            if cleaned:
                lines.append(cleaned)
        return '\n'.join(lines).strip()

    def _stabilize_speakers(self, words: list[_WordState]) -> list[str]:
        if not words:
            return []
        raw_labels = [str(w.speaker or '').strip() for w in words]
        raw_labels = self._collapse_micro_speaker_turns(words, raw_labels)
        stable_speaker = ''
        pending_speaker = ''
        pending_count = 0
        pending_start = 0.0
        out: list[str] = []
        for idx, w in enumerate(words):
            incoming = raw_labels[idx]
            if not incoming:
                out.append(stable_speaker)
                continue
            if not stable_speaker:
                stable_speaker = incoming
                pending_speaker = ''
                pending_count = 0
                out.append(stable_speaker)
                continue
            if incoming == stable_speaker:
                pending_speaker = ''
                pending_count = 0
                out.append(stable_speaker)
                continue
            if incoming == pending_speaker:
                pending_count += 1
            else:
                pending_speaker = incoming
                pending_count = 1
                pending_start = float(w.start)
            pending_duration = max(0.0, float(w.end) - float(pending_start))
            if (
                pending_count >= int(self._speaker_switch_confirm_tokens)
                and pending_duration >= float(self._speaker_switch_min_duration_seconds)
            ):
                stable_speaker = pending_speaker
                pending_speaker = ''
                pending_count = 0
            out.append(stable_speaker)
        return self._collapse_micro_speaker_turns(words, out)

    def _collapse_micro_speaker_turns(self, words: list[_WordState], labels: list[str]) -> list[str]:
        if not words or not labels or len(words) != len(labels):
            self._last_speaker_smoothing_summary = {'micro_turn_merged_count': 0, 'run_count': 0}
            return labels
        runs: list[dict[str, object]] = []
        start_idx = 0
        current = labels[0]
        for idx in range(1, len(labels)):
            if labels[idx] == current:
                continue
            runs.append({'label': current, 'start': start_idx, 'end': idx})
            start_idx = idx
            current = labels[idx]
        runs.append({'label': current, 'start': start_idx, 'end': len(labels)})

        for run in runs:
            start = int(run.get('start') or 0)
            end = int(run.get('end') or start)
            run_words = words[start:end]
            if not run_words:
                run['duration'] = 0.0
                run['chars'] = 0
                continue
            run['duration'] = max(0.0, float(run_words[-1].end) - float(run_words[0].start))
            run['chars'] = sum(len(str(item.word or '').strip()) for item in run_words)
            run['is_micro'] = (
                float(run['duration']) <= float(self._speaker_micro_turn_max_duration_seconds)
                or int(run['chars']) <= int(self._speaker_micro_turn_max_chars)
            )

        out = list(labels)
        merged = 0
        for run_idx, run in enumerate(runs):
            label = str(run.get('label') or '')
            if not label:
                continue
            start = int(run.get('start') or 0)
            end = int(run.get('end') or start)
            if end <= start:
                continue
            duration = float(run.get('duration', 0.0) or 0.0)
            char_count = int(run.get('chars', 0) or 0)
            is_micro = bool(run.get('is_micro', False))
            if not is_micro:
                continue
            prev_run = runs[run_idx - 1] if run_idx > 0 else {}
            next_run = runs[run_idx + 1] if run_idx + 1 < len(runs) else {}
            prev_label = str(prev_run.get('label') or '')
            next_label = str(next_run.get('label') or '')
            replacement = ''
            if prev_label and next_label and prev_label == next_label:
                replacement = prev_label
            else:
                is_tiny = char_count <= 2
                if is_tiny and (prev_label or next_label):
                    if prev_label and not next_label:
                        if int(prev_run.get('chars', 0) or 0) > 2:
                            replacement = prev_label
                    elif prev_label and next_label:
                        prev_score = (
                            int(prev_run.get('chars', 0) or 0),
                            float(prev_run.get('duration', 0.0) or 0.0),
                        )
                        next_score = (
                            int(next_run.get('chars', 0) or 0),
                            float(next_run.get('duration', 0.0) or 0.0),
                        )
                        replacement = prev_label if prev_score >= next_score else next_label
            if not replacement or replacement == label:
                continue
            for idx in range(start, end):
                out[idx] = replacement
            merged += 1
        self._last_speaker_smoothing_summary = {
            'micro_turn_merged_count': int(merged),
            'run_count': int(len(runs)),
            'micro_turn_max_duration_seconds': float(self._speaker_micro_turn_max_duration_seconds),
            'micro_turn_max_chars': int(self._speaker_micro_turn_max_chars),
        }
        return out

    def _speaker_label_to_marker(self, speaker: str) -> str:
        label = str(speaker or '').strip()
        if not label:
            return ''
        if self._speaker_marker_style == 'arrow':
            return '>>'
        return f'[{self._speaker_to_display_label(label).lower()}]'

    def _speaker_to_display_label(self, speaker: str) -> str:
        label = str(speaker or '').strip()
        if not label:
            return ''
        existing = self._speaker_display_map.get(label)
        if existing:
            return existing
        match = re.search(r'(\d+)$', label)
        if match is not None:
            display = f"SPK_{int(match.group(1)):03d}"
        else:
            display = f"SPK_{self._speaker_display_next_index:03d}"
        self._speaker_display_map[label] = display
        self._speaker_display_next_index = max(self._speaker_display_next_index + 1, int(display.rsplit('_', 1)[-1]) + 1)
        return display


    @staticmethod
    def _is_punct(token: str) -> bool:
        return bool(re.fullmatch(r"[\.,!?;:，。！？；：、'\"“”‘’（）()《》〈〉【】\[\]…—\-]+", token or ''))

    @staticmethod
    def _normalize_cjk_token(token: str) -> str:
        if not token:
            return ''
        return re.sub(r'\s+', '', token).strip()

    def _apply_cjk_pause_spacing_to_text(self, text: str, words: list[_WordState]) -> str:
        summary: dict[str, object] = {
            'enabled': bool(self._is_cjk_source),
            'threshold_seconds': float(self._cjk_no_space_gap_seconds),
            'token_count': int(len(words or [])),
            'matched_tokens': 0,
            'pause_spaces': 0,
            'fallback_spaces': 0,
            'fallback_line_breaks': 0,
            'max_gap_seconds': 0.0,
            'reason': '',
        }
        if not self._is_cjk_source:
            summary['reason'] = 'not_cjk'
            self._last_cjk_spacing_summary = summary
            return text
        if not text:
            summary['reason'] = 'empty_text'
            self._last_cjk_spacing_summary = summary
            return text
        if not words:
            spaced = self._insert_cjk_char_fallback_spaces(text, summary=summary)
            summary['reason'] = 'no_tokens'
            self._last_cjk_spacing_summary = summary
            return spaced

        sorted_words = sorted(words, key=lambda w: (w.start, w.end))
        cjk_token_count = sum(1 for word in sorted_words if self._contains_cjk_text(str(word.word or '')))
        if cjk_token_count <= 0:
            summary['reason'] = 'no_cjk_tokens'
            self._last_cjk_spacing_summary = summary
            return text

        compact_chars: list[str] = []
        compact_to_source_index: list[int] = []
        for idx, char in enumerate(text):
            if char.isspace():
                continue
            compact_chars.append(char)
            compact_to_source_index.append(idx)
        compact = ''.join(compact_chars)
        if not compact:
            summary['reason'] = 'empty_compact_text'
            self._last_cjk_spacing_summary = summary
            return text

        insert_before: set[int] = set()
        fallback_before: set[int] = set()
        cursor = 0
        prev: _WordState | None = None
        chars_since_boundary = 0
        for word in sorted_words:
            token = self._normalize_cjk_token(str(word.word or ''))
            if not token:
                continue
            found = compact.find(token, cursor)
            if found < 0:
                continue
            summary['matched_tokens'] = int(summary['matched_tokens']) + 1
            if prev is not None:
                gap = max(0.0, float(word.start) - float(prev.end))
                summary['max_gap_seconds'] = max(float(summary['max_gap_seconds']), gap)
                if gap > self._cjk_no_space_gap_seconds and not self._is_punct(token[:1]):
                    insert_before.add(compact_to_source_index[found])
                    chars_since_boundary = 0
                elif (
                    chars_since_boundary >= self._CJK_MAX_COMPACT_CHARS
                    and self._contains_cjk_text(token)
                    and not self._is_punct(token[:1])
                ):
                    fallback_before.add(compact_to_source_index[found])
                    chars_since_boundary = 0
            chars_since_boundary += len(token)
            cursor = found + len(token)
            prev = word
        if int(summary['matched_tokens']) <= 0:
            spaced = self._insert_cjk_char_fallback_spaces(text, summary=summary)
            summary['reason'] = 'no_token_text_match'
            self._last_cjk_spacing_summary = summary
            return spaced
        insert_before.update(fallback_before)
        if not insert_before:
            summary['reason'] = 'below_threshold'
            self._last_cjk_spacing_summary = summary
            return text

        out: list[str] = []
        for idx, char in enumerate(text):
            if idx in insert_before and out and out[-1] not in {' ', '\n', '\t'}:
                out.append(' ')
            out.append(char)
        summary['pause_spaces'] = sum(1 for idx in insert_before if idx not in fallback_before)
        summary['fallback_spaces'] = len(fallback_before)
        summary['fallback_line_breaks'] = 0
        summary['reason'] = 'inserted'
        self._last_cjk_spacing_summary = summary
        return ''.join(out)

    def _insert_cjk_char_fallback_spaces(self, text: str, *, summary: dict[str, object]) -> str:
        out: list[str] = []
        cjk_count = 0
        inserted = 0
        for char in text:
            is_cjk = self._contains_cjk_text(char)
            if (
                is_cjk
                and cjk_count >= self._CJK_MAX_COMPACT_CHARS
                and out
                and out[-1] not in {' ', '\n', '\t'}
            ):
                out.append(' ')
                inserted += 1
                cjk_count = 0
            out.append(char)
            if is_cjk:
                cjk_count += 1
            elif char.isspace() or self._is_punct(char):
                cjk_count = 0
        summary['fallback_spaces'] = int(summary.get('fallback_spaces', 0) or 0) + inserted
        summary['fallback_line_breaks'] = int(summary.get('fallback_line_breaks', 0) or 0)
        return ''.join(out)

    @staticmethod
    def _contains_cjk_in_words(words: list[_WordState]) -> bool:
        for item in words:
            if SubtitleAssembler._contains_cjk_text(item.word or ''):
                return True
        return False

    @staticmethod
    def _contains_cjk_text(text: str) -> bool:
        return bool(re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', text or ''))

    @staticmethod
    def _words_to_state(words: list[_WordState]) -> list[dict[str, object]]:
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        out: list[dict[str, object]] = []
        for w in ordered:
            out.append({
                'word': w.word,
                'start': float(w.start),
                'end': float(w.end),
                'score': float(w.score),
                'count': int(w.count),
                'last_seen': float(w.last_seen),
                'speaker': str(w.speaker or ''),
            })
        return out

    def _dedupe_words(self, words: list[_WordState]) -> list[_WordState]:
        if not words:
            return []
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        deduped: list[_WordState] = []
        for word in ordered:
            found = self._find_match(deduped, word)
            if found is None:
                deduped.append(word)
            else:
                self._update_word(found, word)
        return deduped

    def _normalize_output_text(self, text: str) -> str:
        if not text:
            return ''
        text = self._normalize_speaker_marker_boundaries(text)
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r'[ \t]+', ' ', line).strip()
            if cleaned:
                lines.append(cleaned)
        return '\n'.join(lines).strip()

    def _merge_by_exact_overlap(self, base: str, incoming: str) -> str:
        base = self._normalize_output_text(base)
        incoming = self._normalize_output_text(incoming)
        if not base:
            return incoming
        if not incoming:
            return base
        overlap = self._max_prefix_suffix_overlap(base, incoming)
        if overlap >= len(incoming):
            return base
        if overlap > 0:
            return self._normalize_output_text(f'{base}{incoming[overlap:]}')
        if incoming in base[-max(16, len(incoming) * 2):]:
            return base
        if self._starts_with_speaker_marker(incoming) and not base.endswith('\n'):
            sep = '\n'
        elif self._should_join_without_space(base, incoming):
            sep = ''
        else:
            sep = '' if base.endswith(('?', '!', ',', '.', ' ', '\n')) else ' '
        return self._normalize_output_text(f'{base}{sep}{incoming}')

    @staticmethod
    def _should_join_without_space(base: str, incoming: str) -> bool:
        left = str(base or '').rstrip()
        right = str(incoming or '').lstrip()
        if not left or not right:
            return False
        if SubtitleAssembler._is_punct(right[0]):
            return True
        return bool(
            re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]$', left)
            or re.search(r'^[\u3400-\u4DBF\u4E00-\u9FFF]', right)
        )

    @staticmethod
    def _starts_with_speaker_marker(text: str) -> bool:
        if not text:
            return False
        return bool(re.match(r'^\s*(?:>>|S\d+:|\[spk_\d+\])\s*', text, flags=re.IGNORECASE))

    @staticmethod
    def _normalize_speaker_marker_boundaries(text: str) -> str:
        if not text:
            return ''
        # Force speaker markers to start on a new line to avoid inline jitter.
        # jitter when overlap-merge glues history + raw chunks.
        marker = r'(>>|S\d+:|\[spk_\d+\])'
        normalized = re.sub(rf'([^\n])\s*{marker}\s*', r'\1\n\2 ', text, flags=re.IGNORECASE)
        normalized = re.sub(rf'^\s*{marker}\s*', r'\1 ', normalized, flags=re.IGNORECASE)
        normalized = re.sub(rf'\n\s*{marker}\s*', r'\n\1 ', normalized, flags=re.IGNORECASE)
        return normalized

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0
