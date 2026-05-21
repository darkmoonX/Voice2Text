"""Subtitle assembly module.

Time-aligned merge strategy:
- `history`: immutable words earlier than current raw window start.
- `stable`: agreed words (count >= agreement_count) still in the active region.
- `partial`: candidate words in the active region.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class _WordState:
    word: str
    start: float
    end: float
    score: float
    count: int
    last_seen: float


class SubtitleAssembler:
    def __init__(self) -> None:
        self._is_cjk_source = False
        self._auto_detect_cjk = True
        self._cjk_no_space_gap_seconds = 0.1
        self.reset()

    def reset(self) -> None:
        self._history_words: list[_WordState] = []
        self._stable_words: list[_WordState] = []
        self._partial_words: list[_WordState] = []
        self._latest_partial_text = ''
        self._last_emitted_source_text = ''
        self._required_agreement_count = 3
        self._score_threshold = 0.60

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

    def merge_incremental_text(self, text: str, *, overlap_merge_method: str, segment_seconds: float, hop_seconds: float, transcription_meta: dict[str, object] | None = None) -> str:
        _ = (overlap_merge_method, segment_seconds, hop_seconds)
        cleaned = re.sub(r'\s+', ' ', text).strip()
        self._latest_partial_text = cleaned
        if not cleaned:
            return ''

        meta = transcription_meta or {}
        elapsed = self._to_float(meta.get('elapsed_seconds', 0.0), 0.0)
        incoming = self._extract_incoming_words(meta, elapsed)
        if incoming and self._auto_detect_cjk:
            self._is_cjk_source = self._contains_cjk_in_words(incoming)

        if incoming:
            raw_start = min((w.start for w in incoming), default=elapsed)
            self._flush_stable_to_history(raw_start)
            self._merge_incoming_words(incoming)
            self._promote_partial_to_stable()
            self._prune_active_words(raw_start)
        else:
            raw_start = elapsed
            self._flush_stable_to_history(raw_start)

        if incoming:
            self._latest_partial_text = self._words_to_text(self._partial_words)
        else:
            self._latest_partial_text = cleaned

        if incoming:
            # When timestamped words exist, build output from word states so
            # CJK spacing policy (cjk_no_space_gap_seconds) remains authoritative.
            all_words = [*self._history_words, *self._stable_words, *self._partial_words]
            merged = self._words_to_text(all_words)
        else:
            # Fallback path for providers/chunks without timestamp metadata.
            history_text = self._words_to_text(self._history_words)
            merged = self._merge_by_exact_overlap(history_text, cleaned)
        merged = self._normalize_output_text(merged)[-1800:]
        if merged == self._last_emitted_source_text:
            return ''
        self._last_emitted_source_text = merged
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
            out.append(_WordState(word=word, start=start_abs, end=end_abs, score=score, count=1, last_seen=end_abs))
        out.sort(key=lambda w: (w.start, w.end))
        return out

    def _merge_incoming_words(self, incoming: list[_WordState]) -> None:
        for word in incoming:
            target = self._find_match(self._stable_words, word)
            if target is not None:
                self._update_word(target, word, increment_count=False)
                continue
            target = self._find_match(self._partial_words, word)
            if target is not None:
                self._update_word(target, word, increment_count=True)
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
    def _update_word(target: _WordState, incoming: _WordState, *, increment_count: bool) -> None:
        target_weight = max(1, int(target.count))
        incoming_weight = max(1, int(incoming.count)) if increment_count else 1
        total_weight = target_weight + incoming_weight
        target.start = ((target.start * target_weight) + (incoming.start * incoming_weight)) / float(total_weight)
        target.end = ((target.end * target_weight) + (incoming.end * incoming_weight)) / float(total_weight)
        target.score = ((target.score * target_weight) + (incoming.score * incoming_weight)) / float(total_weight)
        if increment_count:
            target.count = target_weight + incoming_weight
        target.last_seen = max(target.last_seen, incoming.last_seen)

    def _words_to_text(self, words: list[_WordState]) -> str:
        if not words:
            return ''
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        if not self._is_cjk_source:
            return re.sub(r'\s+', ' ', ' '.join((w.word for w in ordered if w.word))).strip()

        out: list[str] = []
        prev: _WordState | None = None
        for w in ordered:
            token = self._normalize_cjk_token(str(w.word or ''))
            if not token:
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
        return re.sub(r'\s+', ' ', ''.join(out)).strip()


    @staticmethod
    def _is_punct(token: str) -> bool:
        return bool(re.fullmatch(r"[\.,!?;:，。！？；：、'\"“”‘’（）()《》〈〉【】\[\]…—\-]+", token or ''))

    @staticmethod
    def _normalize_cjk_token(token: str) -> str:
        if not token:
            return ''
        return re.sub(r'\s+', '', token).strip()

    @staticmethod
    def _contains_cjk_in_words(words: list[_WordState]) -> bool:
        for item in words:
            if re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', item.word or ''):
                return True
        return False

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
                self._update_word(found, word, increment_count=False)
                found.count = max(found.count, word.count)
        return deduped

    def _normalize_output_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return ''
        return cleaned

    def _merge_by_exact_overlap(self, base: str, incoming: str) -> str:
        base = re.sub(r'\s+', ' ', base).strip()
        incoming = re.sub(r'\s+', ' ', incoming).strip()
        if not base:
            return incoming
        if not incoming:
            return base
        overlap = self._max_prefix_suffix_overlap(base, incoming)
        if overlap >= len(incoming):
            return base
        if overlap > 0:
            return f'{base}{incoming[overlap:]}'.strip()
        if incoming in base[-max(16, len(incoming) * 2):]:
            return base
        sep = '' if base.endswith(('?', '!', ',', '.', ' ')) else ' '
        return f'{base}{sep}{incoming}'.strip()

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0
