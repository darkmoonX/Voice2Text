"""Subtitle assembly module.

Owns incremental merge state so controller stays focused on runtime orchestration.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re


class SubtitleAssembler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._frozen_source_text = ''
        self._active_source_text = ''
        self._last_emitted_source_text = ''
        self._force_sentence_break_once = False

    def merge_incremental_text(self, text: str, *, overlap_merge_method: str, segment_seconds: float, hop_seconds: float) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return ''
        if not self._active_source_text:
            self._active_source_text = cleaned
            if self._force_sentence_break_once and self._frozen_source_text:
                combined = self._concat_with_space(self._frozen_source_text, self._active_source_text)
                self._force_sentence_break_once = False
            else:
                combined = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
            combined = self._normalize_output_text(combined)
            self._last_emitted_source_text = combined
            return combined

        method = self._normalize_merge_method(overlap_merge_method)
        if method == 'commit-on-break':
            combined_prev = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
            combined = self._merge_by_exact_overlap(combined_prev, cleaned)
            combined = self._normalize_output_text(combined)[-1800:]
            if combined == self._last_emitted_source_text:
                return ''
            self._frozen_source_text = ''
            self._active_source_text = combined
            self._last_emitted_source_text = combined
            return combined

        lock_ratio = self._segment_lock_ratio(segment_seconds=segment_seconds, hop_seconds=hop_seconds)
        lock_chars = int(round(len(self._active_source_text) * lock_ratio))
        lock_chars = max(0, min(lock_chars, len(self._active_source_text)))
        lock_chunk = self._active_source_text[:lock_chars].strip()
        overlap_tail = self._active_source_text[lock_chars:].strip()
        if lock_chunk:
            self._frozen_source_text = self._merge_by_exact_overlap(self._frozen_source_text, lock_chunk)

        self._active_source_text = self._merge_stable_tail(overlap_tail, cleaned, lock_ratio)

        combined = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
        combined = self._normalize_output_text(combined)[-1800:]
        if combined == self._last_emitted_source_text:
            return ''
        self._last_emitted_source_text = combined
        return combined

    @staticmethod
    def _normalize_merge_method(raw: str) -> str:
        value = (raw or '').strip().lower()
        if value in {'stable-tail', 'replace-window', 'suffix-overlap', 'fuzzy-overlap'}:
            return 'stable-tail'
        if value in {'commit-on-break', 'append-only'}:
            return 'commit-on-break'
        return 'stable-tail'

    def mark_sentence_break(self) -> None:
        combined = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
        combined = self._normalize_output_text(combined)
        if combined:
            self._frozen_source_text = combined[-1800:]
            self._active_source_text = ''
            self._force_sentence_break_once = True

    @staticmethod
    def _segment_lock_ratio(*, segment_seconds: float, hop_seconds: float) -> float:
        segment = max(0.1, float(segment_seconds))
        hop = max(0.01, float(hop_seconds))
        ratio = hop / segment
        return max(0.05, min(0.95, ratio))

    def _merge_stable_tail(self, overlap_tail: str, incoming: str, lock_ratio: float) -> str:
        previous = re.sub(r'\s+', ' ', overlap_tail).strip()
        latest = re.sub(r'\s+', ' ', incoming).strip()
        if not previous:
            return latest
        if not latest:
            return previous
        confidence = self._front_overlap_confidence(previous, latest)
        confidence_scale = 0.7 + (0.8 * confidence)
        preserve_ratio = max(0.10, min(0.38, lock_ratio * 1.35 * confidence_scale))
        keep_chars = int(round(len(previous) * preserve_ratio))
        min_keep = 4 if self._contains_cjk(previous + latest) else 6
        keep_chars = max(min_keep, min(keep_chars, len(previous)))
        if ' ' in previous and keep_chars < len(previous):
            left_space = previous.rfind(' ', 0, keep_chars)
            right_space = previous.find(' ', keep_chars)
            if left_space >= 0 and keep_chars - left_space <= 6:
                keep_chars = left_space + 1
            elif right_space >= 0 and right_space - keep_chars <= 6:
                keep_chars = right_space + 1
        stable_head = previous[:keep_chars].strip()
        mutable_tail = previous[keep_chars:].strip()
        reconciled_tail = self._merge_by_fuzzy_overlap(mutable_tail, latest)
        if reconciled_tail == latest and mutable_tail:
            exact_tail = self._merge_by_exact_overlap(mutable_tail, latest)
            if exact_tail and exact_tail != mutable_tail:
                reconciled_tail = exact_tail
        merged = self._merge_by_exact_overlap(stable_head, reconciled_tail)
        return merged or previous

    def _front_overlap_confidence(self, previous: str, latest: str) -> float:
        if not previous or not latest:
            return 0.0
        exact = self._max_prefix_suffix_overlap(previous, latest)
        if exact > 0:
            return min(1.0, exact / max(1, min(len(previous), len(latest))))
        probe = min(48, len(previous), len(latest))
        if probe < 4:
            return 0.0
        tail = previous[-probe:]
        head = latest[:probe]
        return SequenceMatcher(None, tail, head).ratio()

    def _compose_rolling_text(self, frozen: str, active: str) -> str:
        return self._merge_by_exact_overlap(frozen, active)

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
        sep = self._join_separator(base, incoming)
        return f'{base}{sep}{incoming}'.strip()

    def _merge_by_fuzzy_overlap(self, base: str, incoming: str) -> str:
        base = re.sub(r'\s+', ' ', base).strip()
        incoming = re.sub(r'\s+', ' ', incoming).strip()
        if not base:
            return incoming
        if not incoming:
            return base
        exact = self._max_prefix_suffix_overlap(base, incoming)
        if exact > 0:
            return self._merge_by_exact_overlap(base, incoming)
        max_len = min(len(base), len(incoming), 120)
        min_len = min(6, max_len)
        best_size = 0
        best_score = 0.0
        for size in range(max_len, min_len - 1, -1):
            tail = base[-size:]
            head = incoming[:size]
            score = SequenceMatcher(None, tail, head).ratio()
            if score > best_score:
                best_score = score
                best_size = size
            if score >= 0.76:
                return f'{base}{incoming[size:]}'.strip()
        if best_size >= 8 and best_score >= 0.62:
            return f'{base}{incoming[best_size:]}'.strip()
        soft_overlap = self._soft_suffix_prefix_overlap(base, incoming)
        if soft_overlap > 0:
            return f'{base}{incoming[soft_overlap:]}'.strip()
        return incoming

    def _normalize_output_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return ''
        deduped = self._collapse_repeated_phrases(cleaned)
        deduped = self._collapse_repeated_char_spans(deduped)
        return re.sub(r'\s+', ' ', deduped).strip()

    def _collapse_repeated_phrases(self, text: str) -> str:
        tokens = [tok for tok in text.split(' ') if tok]
        if len(tokens) >= 6:
            changed = True
            while changed:
                changed = False
                max_ngram = min(14, len(tokens) // 2)
                for size in range(max_ngram, 1, -1):
                    idx = 0
                    while idx + 2 * size <= len(tokens):
                        if tokens[idx:idx + size] == tokens[idx + size:idx + 2 * size]:
                            del tokens[idx + size:idx + 2 * size]
                            changed = True
                        else:
                            idx += 1
                    if changed:
                        break
        merged = ' '.join(tokens) if tokens else text
        return self._collapse_repeated_suffix(merged)

    def _collapse_repeated_suffix(self, text: str) -> str:
        collapsed = text
        for _ in range(3):
            changed = False
            max_size = min(len(collapsed) // 2, 120)
            for size in range(max_size, 6, -1):
                left = collapsed[-2 * size:-size]
                right = collapsed[-size:]
                if not left.strip() or not right.strip():
                    continue
                if SequenceMatcher(None, left, right).ratio() >= 0.9:
                    collapsed = collapsed[:-size].strip()
                    changed = True
                    break
            if not changed:
                break
        return collapsed

    def _collapse_repeated_char_spans(self, text: str) -> str:
        collapsed = text
        for _ in range(3):
            changed = False
            max_size = min(len(collapsed) // 2, 80)
            for size in range(max_size, 2, -1):
                left = collapsed[-2 * size:-size]
                right = collapsed[-size:]
                if not left.strip() or not right.strip():
                    continue
                if left == right:
                    collapsed = collapsed[:-size].strip()
                    changed = True
                    break
                score = SequenceMatcher(None, left, right).ratio()
                if score >= 0.92:
                    collapsed = collapsed[:-size].strip()
                    changed = True
                    break
            if not changed:
                break
        return collapsed

    def _soft_suffix_prefix_overlap(self, base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming), 80)
        if max_len < 3:
            return 0
        min_len = 2 if self._contains_cjk(base + incoming) else 4
        ratio_threshold = 0.78 if self._contains_cjk(base + incoming) else 0.84
        for size in range(max_len, min_len - 1, -1):
            tail = base[-size:]
            head = incoming[:size]
            if SequenceMatcher(None, tail, head).ratio() >= ratio_threshold:
                return size
        return 0

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]', text or ''))

    def _join_separator(self, base: str, incoming: str) -> str:
        if base.endswith(('?', '!', ',', '.', ' ')):
            return ''
        if self._contains_cjk(base[-2:] + incoming[:2]):
            return ''
        return ' '

    @staticmethod
    def _concat_with_space(left: str, right: str) -> str:
        lval = re.sub(r'\s+', ' ', left).strip()
        rval = re.sub(r'\s+', ' ', right).strip()
        if not lval:
            return rval
        if not rval:
            return lval
        return f'{lval} {rval}'.strip()

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0
