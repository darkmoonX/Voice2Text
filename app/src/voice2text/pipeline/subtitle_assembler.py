"""Subtitle assembly module.

Time-aligned merge strategy:
- `history`: immutable words earlier than current raw window start.
- `stable`: agreed words (count >= agreement_count) still in the active region.
- `partial`: candidate words in the active region.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import time
from typing import Callable


@dataclass
class _WordState:
    word: str
    start: float
    end: float
    score: float
    count: int
    last_seen: float
    speaker: str = ''
    # On-time per-window diarization label (kept alongside the profile-preferred
    # `speaker`). Unused by rendering today; retained so a future speaker re-anchor
    # can find a turn's true onset (local flips ~on-time; profile identity lags).
    local_speaker: str = ''


# Round 0052 Phase B: sentinel a relabel resolver returns when its (asynchronous) resolution for
# the requested span is not ready yet -- the assembler keeps the batch pending and retries on a
# later drain instead of freezing it with unrefined labels.
RELABEL_PENDING = object()


class SubtitleAssembler:
    _CJK_MAX_COMPACT_CHARS = 18
    _HISTORY_TAIL_DEDUPE_WORDS = 160
    _DEDUPE_BUCKET_SECONDS = 0.5
    # Visual rule drawn between fully-committed history and the live raw window in
    # the overlay-only frame (round 0017). Display-only: never enters word state,
    # merge keys, the export-facing source text, or the CER comparison.
    _LIVE_RAW_SEPARATOR = '┄' * 12

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
        self._rolling_visible_text = ''
        self._rolling_committed_text = ''
        self._rolling_committed_last_speaker = ''
        # Absolute end time of the last word folded into the committed text, so a
        # long same-speaker pause that straddles a commit boundary still breaks.
        self._rolling_committed_last_end: float | None = None
        # Delayed-freeze buffer: stable words that have been flushed out of the
        # active window but are held (still as word states, re-rendered each frame)
        # before being baked into the frozen committed string. This lets a late
        # cross-window profile identity (warmup ~24s) back-fill the speaker on a new
        # turn's onset words while they are still mutable, so the marker freezes at
        # the true onset instead of where the profile finally confirmed. Empty when
        # commit-hold is disabled (the words go straight to the frozen string).
        # Held as whole flush BATCHES (not a flat list): each batch is drained via the
        # same per-flush `_append_to_rolling_committed_text` the legacy path uses, so
        # the committed text stays byte-identical to legacy except for the re-anchored
        # speaker markers (a flat buffer re-merged at arbitrary cut points diverges).
        self._pending_commit_batches: list[list[_WordState]] = []
        # End-of-stream finalize returns the longest clean merged snapshot seen
        # (the last "full" window before the audio tails off). That snapshot is a
        # real overlay frame the hot path already produced and emitted, so it
        # recovers the trailing words that never reached the stable-promotion
        # agreement count without dumping the raw partial-word buffer (which is
        # character-interleaved across the overlapping windows that contributed
        # it) and without re-running any cross-window merge that could resurrect
        # the duplication classes 0003/0004 fixed.
        self._finalize_snapshot_text = ''
        self._finalize_snapshot_len = 0
        # Overlay-only decorated frame (committed history | separator | raw window
        # with an immediate speaker marker). Recomputed each non-empty merge; read
        # via get_live_overlay_frame(). Kept separate from the returned clean text.
        self._last_overlay_frame = ''
        self._speaker_display_map: dict[str, str] = {}
        self._speaker_display_next_index = 0
        self._required_agreement_count = 3
        # Word-confidence gate before cross-window agreement. wav2vec2 alignment
        # scores are on different scales per language: CJK alignment is highly
        # confident (median ~0.96), but English/Latin alignment is systematically
        # lower (median ~0.37). A single CJK-calibrated 0.60 cut drops ~85% of
        # legitimate English words before they can reach the agreement count,
        # collapsing the realtime English transcript. Keep 0.60 for CJK and use a
        # much lower gate for non-CJK so each language drops only its genuine
        # ~0-score alignment failures (CJK keeps ~95%, English keeps ~90%).
        self._score_threshold = 0.60
        self._score_threshold_non_cjk = 0.10
        # Non-CJK word-match tolerances (see `_word_matches`): loose enough to
        # merge jittery English alignment copies of one spoken word into a single
        # accumulating state. IoU disabled (0.0) because English word intervals
        # frequently fail to overlap across windows.
        self._alignment_enabled = True
        self._non_cjk_match_start_diff = 1.0
        self._non_cjk_match_end_diff = 1.1
        self._non_cjk_match_min_iou = 0.0
        self._speaker_switch_confirm_tokens = 2
        self._speaker_switch_min_duration_seconds = 0.18
        self._speaker_micro_turn_max_duration_seconds = 0.80
        self._speaker_micro_turn_max_chars = 4
        # When the same speaker resumes after a silence longer than this, re-emit
        # the marker behind a blank-line hard boundary so a long pause reads as a
        # new utterance. Driven off absolute word times (works across the
        # overlapping STT windows where provider-side, window-relative detection
        # cannot). 0 disables.
        self._speaker_pause_break_seconds = 1.8
        # Delayed-freeze (speaker re-anchor) controls. `commit_hold_seconds` = how
        # long a flushed word is held in `_pending_commit_words` before it is baked
        # into the frozen committed string; it must cover the profile-warmup lag for
        # the late identity to arrive (~24s visible threshold) plus margin. 0.0
        # disables the hold entirely (flushed words freeze immediately = legacy
        # byte-identical path). Stabilization criterion for confirming/holding a
        # pending turn boundary: 'consecutive' mirrors the live gate (N same-label
        # words spanning a min duration); 'majority' confirms when a label holds
        # >= ratio of a trailing window (better for interleaved Q&A). Tunable.
        self._commit_hold_seconds = 0.0
        self._reanchor_stabilization = 'consecutive'
        self._reanchor_majority_window_seconds = 2.0
        self._reanchor_majority_min_ratio = 0.6
        # Round 0048: pre-commit local-diarization relabel resolver, injected by the loop via
        # `set_relabel_resolver`. Called once per pending batch, right before it freezes (inside
        # `_drain_pending_commits`), with the batch's absolute (start, end) span. Returns either
        # a resolved speaker id (str, legacy whole-batch overwrite), a list of labeled span dicts
        # ({"start","end","resolved","resolved_cosine","scores"} -- round 0052 turn-aware
        # per-word apply with the margin gate), or None (no confident relabel / disabled / any
        # failure -> no-op, keep existing labels). None resolver = feature off, byte-identical.
        self._relabel_resolver: Callable[[float, float], object] | None = None
        # Round 0052 margin gate: a resolved profile only overwrites a word's existing non-empty
        # label when its cosine beats the incumbent label's own cosine by this margin.
        self._relabel_margin = 0.05
        # Round 0052 Phase B: when True the resolver is asynchronous -- sentence breaks route
        # stable words through the pending buffer (instead of force-draining + direct append) so
        # batches can wait for their worker resolution while commit order stays monotonic.
        self._relabel_defer_enabled = False
        # Final display-script fold (one consistent Simplified/Traditional script
        # in the visible/exported text). '' disables. Applied only to the output
        # projection -- never to internal word state or the overlap-comparison
        # keys -- so dedup/merge and CER stay byte-neutral (char-level s2t/t2s).
        self._display_script = ''
        self._last_cjk_spacing_summary: dict[str, object] = {}
        self._last_speaker_smoothing_summary: dict[str, object] = {}
        self._last_merge_diagnostics: dict[str, object] = {}
        self._last_overlap_summary: dict[str, object] = {}
        self._last_history_dedupe_summary: dict[str, object] = {}

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

    def set_speaker_pause_break_seconds(self, seconds: float) -> None:
        try:
            value = float(seconds)
        except Exception:
            value = 1.8
        self._speaker_pause_break_seconds = max(0.0, value)

    def set_commit_hold(
        self,
        *,
        hold_seconds: float | None = None,
        stabilization: str | None = None,
        majority_window_seconds: float | None = None,
        majority_min_ratio: float | None = None,
    ) -> None:
        """Configure delayed-freeze speaker re-anchoring. `hold_seconds<=0` keeps the
        legacy immediate-freeze path (byte-identical)."""
        if hold_seconds is not None:
            try:
                self._commit_hold_seconds = max(0.0, float(hold_seconds))
            except Exception:
                self._commit_hold_seconds = 0.0
        if stabilization is not None:
            token = str(stabilization or '').strip().lower()
            self._reanchor_stabilization = 'majority' if token == 'majority' else 'consecutive'
        if majority_window_seconds is not None:
            try:
                self._reanchor_majority_window_seconds = max(0.1, float(majority_window_seconds))
            except Exception:
                pass
        if majority_min_ratio is not None:
            try:
                self._reanchor_majority_min_ratio = max(0.0, min(1.0, float(majority_min_ratio)))
            except Exception:
                pass

    def set_relabel_resolver(
        self,
        resolver: Callable[[float, float], object] | None,
        *,
        margin: float | None = None,
        defer: bool | None = None,
    ) -> None:
        """Round 0048: inject the pre-commit local-diarization relabel resolver. `None` disables
        the feature (default) -- `_drain_pending_commits` skips the relabel call entirely.
        `margin` (round 0052) configures the turn-aware overwrite gate; `defer` (Phase B) marks
        the resolver as asynchronous (may return RELABEL_PENDING; sentence breaks route through
        the pending buffer). None keeps the current value."""
        self._relabel_resolver = resolver
        if margin is not None:
            try:
                self._relabel_margin = max(0.0, min(1.0, float(margin)))
            except Exception:
                pass
        if defer is not None:
            self._relabel_defer_enabled = bool(defer)

    def set_display_script(self, script: str | None) -> None:
        token = str(script or '').strip().lower()
        if token in {'hant', 'zh-hant', 'zh-tw', 'zh-hk', 'tw', 'traditional'}:
            self._display_script = 'hant'
        elif token in {'hans', 'zh-hans', 'zh-cn', 'zh-sg', 'cn', 'simplified'}:
            self._display_script = 'hans'
        else:
            self._display_script = ''

    def _project_display_script(self, text: str) -> str:
        """Fold the final output to one display script (char-level, no vocab change)."""
        if not text or not self._display_script:
            return text
        from voice2text.stt.audio_utils import unify_chinese_script
        return unify_chinese_script(text, self._display_script)

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
            'history_word_count': int(len(self._history_words)),
            'rolling_base_chars': 0,
            'visible_chars': 0,
            'merged_chars': 0,
            'visible_source': 'rolling',
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
        self._alignment_enabled = bool(meta.get('alignment_enabled', True))
        step_started_at = time.perf_counter()
        incoming = self._extract_incoming_words(meta, elapsed)
        diagnostics['extract_seconds'] = time.perf_counter() - step_started_at
        diagnostics['incoming_count'] = int(len(incoming))
        if incoming and self._auto_detect_cjk:
            self._is_cjk_source = self._contains_cjk_in_words(incoming)

        step_started_at = time.perf_counter()
        if incoming:
            raw_start = min((w.start for w in incoming), default=elapsed)
            moved_to_history = self._flush_stable_to_history(raw_start)
            self._absorb_moved_into_commit(moved_to_history)
            self._merge_incoming_words(incoming)
            self._promote_partial_to_stable()
            self._prune_active_words(raw_start, retention_seconds=segment_seconds)
            self._reanchor_and_drain_pending(now=elapsed)
        else:
            raw_start = elapsed
            moved_to_history = self._flush_stable_to_history(raw_start)
            self._absorb_moved_into_commit(moved_to_history)
            self._reanchor_and_drain_pending(now=elapsed)
        diagnostics['state_update_seconds'] = time.perf_counter() - step_started_at
        diagnostics['history_dedupe'] = dict(self._last_history_dedupe_summary)
        diagnostics['moved_to_history_count'] = int(len(moved_to_history))
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

        step_started_at = time.perf_counter()
        visible_cleaned = self._apply_cjk_pause_spacing_to_text(cleaned, incoming)
        diagnostics['spacing_seconds'] = time.perf_counter() - step_started_at
        diagnostics['visible_chars'] = int(len(visible_cleaned))
        # UI rolling output is separate from full immutable history. Stable
        # words that moved into history are appended once to a committed UI
        # buffer; each new raw window replaces the previous raw tail through
        # overlap merge. This models the overlay as: committed history + raw.
        committed_base = self._committed_base_text()
        rolling_base = committed_base or self._rolling_visible_text
        rolling_base_source = 'committed_history' if committed_base else 'previous_rolling'
        diagnostics['rolling_base_source'] = rolling_base_source
        diagnostics['history_render_seconds'] = 0.0
        diagnostics['history_chars'] = 0
        diagnostics['history_word_count'] = int(len(self._history_words))
        diagnostics['rolling_committed_chars'] = int(len(self._rolling_committed_text))
        diagnostics['rolling_base_chars'] = int(len(rolling_base))
        step_started_at = time.perf_counter()
        merged = self._merge_by_exact_overlap(rolling_base, visible_cleaned)
        diagnostics['overlap_seconds'] = time.perf_counter() - step_started_at
        diagnostics['rolling_overlap'] = dict(self._last_overlap_summary)
        step_started_at = time.perf_counter()
        merged = self._normalize_output_text(merged)
        diagnostics['final_normalize_seconds'] = time.perf_counter() - step_started_at
        diagnostics['merged_chars'] = int(len(merged))
        self._rolling_visible_text = merged
        # Retain the longest clean merged snapshot for end-of-stream finalize.
        if len(merged) > self._finalize_snapshot_len:
            self._finalize_snapshot_len = len(merged)
            self._finalize_snapshot_text = merged
        diagnostics['history_count_after'] = int(len(self._history_words))
        diagnostics['stable_count_after'] = int(len(self._stable_words))
        diagnostics['partial_count_after'] = int(len(self._partial_words))
        if merged == self._last_emitted_source_text:
            diagnostics['returned_empty'] = True
            diagnostics['total_seconds'] = time.perf_counter() - total_started_at
            self._last_merge_diagnostics = diagnostics
            return ''
        self._last_emitted_source_text = merged
        self._last_overlay_frame = self._compose_live_overlay_frame(merged, incoming)
        diagnostics['total_seconds'] = time.perf_counter() - total_started_at
        self._last_merge_diagnostics = diagnostics
        return self._project_display_script(merged)

    def _extract_incoming_words(self, meta: dict[str, object], elapsed: float) -> list[_WordState]:
        items = meta.get('token_timestamps')
        if not isinstance(items, list):
            return []
        # Pick the confidence gate by script before filtering. The persistent
        # `_is_cjk_source` is only updated by add_window *after* this extract, so
        # in auto-detect mode decide from this batch's own words to avoid a
        # one-window lag (and a wrong gate on the very first window).
        if self._auto_detect_cjk:
            batch_is_cjk = self._items_contain_cjk(items)
        else:
            batch_is_cjk = self._is_cjk_source
        score_threshold = self._score_threshold if batch_is_cjk else self._score_threshold_non_cjk
        out: list[_WordState] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            word = str(raw.get('word') or '').strip()
            if not word:
                continue
            score = self._to_float(raw.get('score'), 0.0)
            if score < score_threshold:
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
                    speaker=str(raw.get('speaker') or raw.get('profile_speaker') or '').strip(),
                    local_speaker=str(raw.get('local_speaker') or '').strip(),
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

    def _flush_stable_to_history(self, raw_start: float) -> list[_WordState]:
        if not self._stable_words:
            self._set_history_dedupe_summary('none', 0, len(self._history_words), len(self._history_words))
            return []
        remain: list[_WordState] = []
        moved: list[_WordState] = []
        for word in self._stable_words:
            if word.end <= raw_start:
                moved.append(word)
            else:
                remain.append(word)
        self._stable_words = remain
        if moved:
            self._append_history_words(moved)
        else:
            self._set_history_dedupe_summary('none', 0, len(self._history_words), 0)
        return moved

    def _append_history_words(self, words: list[_WordState]) -> None:
        if not words:
            self._set_history_dedupe_summary('none', 0, len(self._history_words), 0)
            return
        moved = self._dedupe_words(words)
        if not moved:
            self._set_history_dedupe_summary('none', 0, len(self._history_words), 0)
            return
        if not self._history_words:
            self._history_words = moved
            self._set_history_dedupe_summary('init', len(moved), len(self._history_words), 0, moved_count=len(moved))
            return

        moved.sort(key=lambda w: (w.start, w.end))
        history_count_before = len(self._history_words)
        tail_size = max(self._HISTORY_TAIL_DEDUPE_WORDS, len(moved) * 4)
        split_at = max(0, history_count_before - tail_size)
        prefix = self._history_words[:split_at]
        tail = self._history_words[split_at:]

        # Normal streaming input appends near the history tail. If an older
        # correction arrives before the bounded tail, fall back to full dedupe.
        if prefix and moved[0].start < prefix[-1].start:
            combined = self._history_words + moved
            self._history_words = self._dedupe_words(combined)
            self._set_history_dedupe_summary(
                'full',
                len(combined),
                len(self._history_words),
                history_count_before,
                moved_count=len(moved),
                tail_count=len(tail),
            )
            return

        merged_tail = self._dedupe_words(tail + moved)
        self._history_words = prefix + merged_tail
        self._set_history_dedupe_summary(
            'tail',
            len(tail) + len(moved),
            len(self._history_words),
            history_count_before,
            moved_count=len(moved),
            tail_count=len(tail),
            prefix_count=len(prefix),
        )

    def _set_history_dedupe_summary(
        self,
        mode: str,
        input_count: int,
        history_count: int,
        history_count_before: int,
        *,
        moved_count: int = 0,
        tail_count: int = 0,
        prefix_count: int = 0,
    ) -> None:
        self._last_history_dedupe_summary = {
            'mode': str(mode),
            'input_count': int(input_count),
            'history_count': int(history_count),
            'history_count_before': int(history_count_before),
            'moved_count': int(moved_count),
            'tail_count': int(tail_count),
            'prefix_count': int(prefix_count),
        }

    def _prune_active_words(self, raw_start: float, *, retention_seconds: float = 1.0) -> None:
        # Keep unresolved words within the active audio window. The old 1s
        # cutoff could prune correct end-of-stream tail words before EOF
        # finalize had a chance to commit them.
        try:
            retention = float(retention_seconds)
        except Exception:
            retention = 1.0
        cutoff = raw_start - max(1.0, retention)
        self._partial_words = [w for w in self._partial_words if w.end >= cutoff]
        self._partial_words.sort(key=lambda w: (w.start, w.end))

    def finalize(self) -> str:
        # Drain any delayed-freeze pending words (with their final re-anchored
        # speakers) into the frozen committed string before snapshotting (no-op when
        # commit-hold is disabled).
        self._reanchor_and_drain_pending(now=float('inf'), force_all=True)
        # Flush still-unresolved words into the immutable history buffer so the
        # full record stays complete for export/debug.
        pending = list(self._stable_words) + list(self._partial_words)
        finalized_word_count = len(pending)
        if pending:
            deduped = self._dedupe_words(pending)
            if deduped:
                self._append_history_words(deduped)
        self._stable_words = []
        self._partial_words = []
        self._latest_partial_text = ''

        # Return the longest clean merged snapshot, not the raw partial-word
        # buffer. The partial buffer holds the same trailing phrase from several
        # overlapping windows at slightly offset timestamps; rendering it
        # character-interleaves the scripts (the bug class 0003/0004 fixed for
        # the hot path). The snapshot already went through the full overlap
        # ladder when it was the live window, so it stays clean and also carries
        # any tail that the final tail-off windows dropped from the rolling view.
        snapshot = self._finalize_snapshot_text
        rolling = self._rolling_visible_text
        candidate = snapshot if len(snapshot) >= len(rolling) else rolling
        prev_committed = self._rolling_committed_text
        final_text = self._normalize_output_text(candidate or self.get_history_text())
        self._rolling_committed_text = final_text
        self._rolling_visible_text = final_text
        self._last_merge_diagnostics = {
            'finalized': True,
            'finalized_word_count': int(finalized_word_count),
            'finalize_snapshot_chars': int(len(snapshot)),
            'history_count_after': int(len(self._history_words)),
            'stable_count_after': 0,
            'partial_count_after': 0,
            'history_dedupe': dict(self._last_history_dedupe_summary),
            'merged_chars': int(len(final_text)),
        }
        if not final_text:
            return ''
        # Emit only when finalize actually changed state: either it flushed
        # un-committed words, or the recovered snapshot differs from what was
        # already committed. Repeated finalize() calls and no-op finalizes
        # (everything already in history) stay silent.
        if finalized_word_count <= 0 and final_text == prev_committed:
            return ''
        self._last_emitted_source_text = final_text
        return self._project_display_script(final_text)

    def mark_sentence_break(self, now: float | None = None) -> None:
        # Force current confirmed words into immutable history on sentence break.
        # Drain any older held pending words first so commit order stays monotonic.
        if self._relabel_resolver is not None and self._relabel_defer_enabled and self._commit_hold_seconds > 0.0:
            # Round 0052 Phase B: an async resolver may not have this burst's resolutions yet --
            # don't force-freeze past it. Drain whatever has actually aged past the hold window
            # (real `now`, not inf -- inf would force-drain everything and defeat the point of
            # holding), then route the stable words through the pending buffer as one batch (same
            # FIFO, so commit order stays monotonic while the batch waits on its worker).
            effective_now = float(now) if now is not None else float(self._rolling_committed_last_end or 0.0)
            self._reanchor_and_drain_pending(now=effective_now, force_all=False)
            if self._stable_words:
                moved = list(self._stable_words)
                self._append_history_words(moved)
                self._stable_words = []
                self._absorb_moved_into_commit(moved)
            return
        self._reanchor_and_drain_pending(now=float('inf'), force_all=True)
        if self._stable_words:
            moved = list(self._stable_words)
            self._append_history_words(moved)
            self._stable_words = []
            self._append_to_rolling_committed_text(moved)

    def get_stable_text(self) -> str:
        return self._words_to_text(self._stable_words)

    def get_partial_text(self) -> str:
        return self._latest_partial_text

    def get_live_overlay_frame(self) -> str:
        """Overlay-only source frame: committed history | separator | live raw window.

        Display-only decoration of the most recent merge. The clean text returned by
        merge_incremental_text (used for translation + transcript export) is never
        decorated, so export/CER stay byte-identical.
        """
        return self._project_display_script(self._last_overlay_frame)

    def _compose_live_overlay_frame(self, merged: str, incoming: list[_WordState]) -> str:
        # Raw region text comes from the CLEAN deduped `merged` (never the raw word
        # state, which is char-interleaved across overlapping windows). The split
        # point is where committed history ends inside merged: exact prefix in the
        # dominant paths, longest-common-prefix as a robust fallback when a marker
        # re-anchor revised the committed tail.
        committed = self._normalize_output_text(self._committed_base_text())
        if not committed:
            return merged  # nothing fully fixed yet -> all live
        if merged.startswith(committed):
            boundary = len(committed)
        else:
            boundary = len(self._common_prefix(merged, committed))
        raw_region = merged[boundary:].strip()
        if not raw_region:
            return merged  # no live edge beyond committed -> no rule this frame
        committed_part = merged[:boundary].rstrip() or committed
        raw_region = self._prefix_immediate_speaker(raw_region, incoming)
        return f'{committed_part}\n{self._LIVE_RAW_SEPARATOR}\n{raw_region}'

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return a[:idx]

    def _prefix_immediate_speaker(self, raw_region: str, incoming: list[_WordState]) -> str:
        # One immediate (un-gated) marker for the live region: the most recent
        # window word's raw diarization label. Flips the moment a new speaker takes
        # the live edge, while committed history keeps its stable gated markers.
        if not raw_region or self._starts_with_speaker_marker(raw_region):
            return raw_region
        speaker = ''
        for word in sorted(incoming, key=lambda w: (w.start, w.end), reverse=True):
            token = str(word.speaker or '').strip()
            if token:
                speaker = token
                break
        marker = self._speaker_label_to_marker(speaker) if speaker else ''
        if not marker:
            return raw_region
        return f'{marker} {raw_region}'


    def get_history_text(self) -> str:
        return self._words_to_text(self._history_words)

    def get_history_tail_text(self, max_words: int = 160) -> str:
        try:
            limit = max(1, int(max_words))
        except Exception:
            limit = 160
        return self._words_to_text(self._history_words[-limit:])

    def get_prompt_tail(self, max_chars: int = 160) -> str:
        """Recent committed text as a decode prompt: markers/newlines stripped.

        Uses the committed (stable) rolling text only -- never the volatile raw
        window -- so a per-window initial_prompt is conditioned on already-agreed
        context, bounding prompt-feedback risk. Returns '' when disabled.
        """
        try:
            limit = int(max_chars)
        except Exception:
            limit = 0
        if limit <= 0:
            return ''
        base = self._rolling_committed_text or self._rolling_visible_text or ''
        if not base:
            return ''
        stripped = re.sub(r'\[spk_\d+\]|>>', ' ', base)
        stripped = re.sub(r'\s+', ' ', stripped).strip()
        if not stripped:
            return ''
        return stripped[-limit:]

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
            'rolling_visible_chars': len(self._rolling_visible_text),
            'rolling_committed_chars': len(self._rolling_committed_text),
            'is_cjk_source': bool(self._is_cjk_source),
            'auto_detect_cjk': bool(self._auto_detect_cjk),
            'cjk_spacing': dict(self._last_cjk_spacing_summary),
            'speaker_smoothing': dict(self._last_speaker_smoothing_summary),
            'history_dedupe': dict(self._last_history_dedupe_summary),
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
        key = word.strip().lower().replace(' ', '')
        if not key:
            return ''
        try:
            from voice2text.stt.audio_utils import normalize_chinese_script

            return normalize_chinese_script(key, 'hans')
        except Exception:
            return key

    @staticmethod
    def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        return (inter / union) if union > 0 else 0.0

    def _word_matches(
        self,
        a: _WordState,
        b: _WordState,
        *,
        max_start_diff: float | None = None,
        max_end_diff: float | None = None,
        min_iou: float | None = None,
    ) -> bool:
        # Tolerances are script- and alignment-aware. CJK forced alignment is
        # tight and stable, so the same word in adjacent windows lands within
        # ~0.25s with high interval overlap; round 0063 found CJK without forced
        # alignment can jitter like the loose-tolerance cases. English/Latin
        # wav2vec2 alignment jitters far more (up to ~1s) and its word intervals
        # often barely overlap, so the tight CJK gate fails to merge the jittered
        # copies of one spoken word: at agreement count 3 they each under-confirm
        # (dropped words), and if promoted they render as adjacent duplicates
        # (`completely completely`). A loose non-CJK gate with IoU disabled merges
        # those copies into one accumulating state -> complete AND duplicate-free.
        # (Replay: vskw 71%/9dup -> 95%/0dup; mdqm 72% -> 86%.)
        use_tight_cjk_tolerance = self._is_cjk_source and self._alignment_enabled
        if max_start_diff is None:
            max_start_diff = 0.25 if use_tight_cjk_tolerance else self._non_cjk_match_start_diff
        if max_end_diff is None:
            max_end_diff = 0.35 if use_tight_cjk_tolerance else self._non_cjk_match_end_diff
        if min_iou is None:
            min_iou = 0.30 if use_tight_cjk_tolerance else self._non_cjk_match_min_iou
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
        # Keep the on-time local label; fill if the target never had one. (Local labels
        # are per-window; the earliest sighting is the most on-time, so don't overwrite.)
        if (not target.local_speaker) and incoming.local_speaker:
            target.local_speaker = incoming.local_speaker

    def _words_to_text(self, words: list[_WordState], *, initial_speaker_label: str = '') -> str:
        if not words:
            return ''
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        speaker_labels = self._stabilize_speakers(ordered)
        out: list[str] = []
        prev: _WordState | None = None
        last_speaker_label = str(initial_speaker_label or '').strip()
        for (w, speaker) in zip(ordered, speaker_labels):
            raw_token = str(w.word or '')
            token = self._normalize_cjk_token(raw_token) if self._is_cjk_source else raw_token.strip()
            if not token:
                continue
            marker = self._speaker_label_to_marker(speaker)
            speaker_changed = bool(marker and speaker and (speaker != last_speaker_label))
            # Gap to the previous rendered word. A long same-speaker gap inside
            # this batch is a pause break; a speaker change already starts its own
            # line. Pauses across a commit boundary are handled by
            # _append_to_rolling_committed_text (the leading break here would be
            # stripped/merged away).
            inter_gap = (float(w.start) - float(prev.end)) if prev is not None else None
            pause_break = (
                not speaker_changed
                and bool(marker and speaker)
                and speaker == last_speaker_label
                and float(self._speaker_pause_break_seconds) > 0.0
                and inter_gap is not None
                and inter_gap > float(self._speaker_pause_break_seconds)
            )
            if speaker_changed:
                if self._is_cjk_source:
                    out.append(f'\n{marker} ')
                else:
                    out.append(f'\n{marker}' if out else f'{marker}')
                last_speaker_label = speaker
            elif pause_break:
                # Blank-line hard boundary so _collapse_redundant_speaker_marker_lines
                # keeps the re-emitted same-speaker marker instead of collapsing it.
                if self._is_cjk_source:
                    out.append(f'\n\n{marker} ')
                else:
                    out.append(f'\n\n{marker}')
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
                elif lines and lines[-1] != '':
                    lines.append('')
            return '\n'.join(lines).strip()
        text = ''.join(out)
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r'[ \t]+', ' ', line).strip()
            if cleaned:
                lines.append(cleaned)
            elif lines and lines[-1] != '':
                lines.append('')
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
                is_short_island = (
                    char_count <= int(self._speaker_micro_turn_max_chars)
                    and duration <= float(self._speaker_micro_turn_max_duration_seconds)
                )
                if is_short_island and (prev_label or next_label):
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

    def apply_speaker_alias_remap(self, remap: dict[str, str] | None) -> None:
        """Round 0053: when the profile store merges/reconciles ids (`{dropped_id: kept_id}`,
        e.g. the shipped `whisperx_speaker_profile_reconcile_threshold` auto-reconcile, or a
        future direct-informed consolidation), keep the DISPLAYED speaker number continuous
        instead of letting the surviving id acquire a fresh one. This is the exact mechanism
        behind round 0046's churn: words already emitted under `dropped_id` carry an established
        display alias; future windows assign `kept_id` instead, which -- without this hook --
        gets its OWN fresh display number on first sight, making one continuing speaker appear to
        split. If either side of the remap already has a display alias, both ids point to that
        same alias going forward. Never rewrites already-emitted text (forward-only, alias
        resolution only touches how FUTURE words with these raw ids render); safe to call every
        window with an empty/no-op remap (byte-identical when reconcile doesn't fire)."""
        if not remap:
            return
        for (old_id, new_id) in remap.items():
            old_token = str(old_id or '').strip()
            new_token = str(new_id or '').strip()
            if not old_token or not new_token or old_token == new_token:
                continue
            alias = self._speaker_display_map.get(old_token) or self._speaker_display_map.get(new_token)
            if not alias:
                continue
            self._speaker_display_map[old_token] = alias
            self._speaker_display_map[new_token] = alias


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
    def _items_contain_cjk(items: list) -> bool:
        """CJK presence over raw token_timestamps dicts (pre-extraction)."""
        for raw in items:
            if isinstance(raw, dict) and SubtitleAssembler._contains_cjk_text(str(raw.get('word') or '')):
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
        index: dict[tuple[str, int], list[int]] = {}
        for word in ordered:
            norm = self._normalize_word(word.word)
            bucket = self._dedupe_bucket(word.start)
            found: _WordState | None = None
            for nearby_bucket in range(bucket - 1, bucket + 2):
                for candidate_index in index.get((norm, nearby_bucket), []):
                    candidate = deduped[candidate_index]
                    if self._word_matches(candidate, word):
                        found = candidate
                        break
                if found is not None:
                    break
            if found is None:
                deduped.append(word)
                index.setdefault((norm, bucket), []).append(len(deduped) - 1)
            else:
                self._update_word(found, word)
        return deduped

    @classmethod
    def _dedupe_bucket(cls, start: float) -> int:
        try:
            return int(float(start) / cls._DEDUPE_BUCKET_SECONDS)
        except Exception:
            return 0

    def _append_to_rolling_committed_text(self, words: list[_WordState]) -> None:
        if not words:
            return
        text, last_speaker, last_end = self._fold_words_into_committed(
            self._rolling_committed_text,
            self._rolling_committed_last_speaker,
            self._rolling_committed_last_end,
            words,
        )
        self._rolling_committed_text = text
        self._rolling_committed_last_speaker = last_speaker
        self._rolling_committed_last_end = last_end

    def _fold_words_into_committed(
        self,
        committed_text: str,
        last_speaker: str,
        last_end: float | None,
        words: list[_WordState],
    ) -> tuple[str, str, float | None]:
        """Pure fold of one flush batch into a committed string. Returns the new
        (text, last_speaker, last_end) without mutating instance commit state, so the
        delayed-freeze preview can fold the held batches non-destructively and match
        exactly what the eventual per-batch drain produces."""
        moved_text = self._words_to_text(words, initial_speaker_label=last_speaker)
        if not moved_text:
            return committed_text, last_speaker, last_end
        previous_summary = dict(self._last_overlap_summary)
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        first_speaker = ''
        for label in self._stabilize_speakers(ordered):
            token = str(label or '').strip()
            if token:
                first_speaker = token
                break
        # A long same-speaker silence that straddles this commit boundary (the
        # words after the pause arrive in a later window than the words before
        # it) still reads as a new utterance: append behind a blank-line hard
        # boundary + re-emitted marker instead of overlap-merging inline.
        boundary_pause = (
            bool(committed_text)
            and last_end is not None
            and float(self._speaker_pause_break_seconds) > 0.0
            and bool(first_speaker)
            and first_speaker == last_speaker
            and (float(ordered[0].start) - float(last_end)) > float(self._speaker_pause_break_seconds)
        )
        if boundary_pause:
            marker = self._speaker_label_to_marker(first_speaker)
            new_text = self._normalize_output_text(f'{committed_text}\n\n{marker} {moved_text}')
        else:
            base = committed_text
            merged = self._merge_by_exact_overlap(base, moved_text)
            # Collapse guard: a committed-building merge must never drop more than
            # the incoming chunk's worth of prior committed text. The marker-leading
            # merge path can over-match a new speaker's [spk_new] chunk against the
            # previous speaker's committed tail and delete it (round 0019: committed
            # 302->46 chars at a transition under aggressive assignment). When that
            # happens, fall back to a plain append (no cross-boundary dedupe). The
            # guard is inert for sane merges, so normal CER is unchanged.
            if base and (len(base) - len(merged)) > len(moved_text):
                sep = '' if base.endswith('\n') else '\n'
                merged = self._normalize_output_text(f'{base}{sep}{moved_text}')
            new_text = merged
        self._last_overlap_summary = previous_summary
        new_last_speaker = self._last_non_empty_speaker_label(words) or last_speaker
        committed_end = max((float(w.end) for w in words), default=None)
        new_last_end = committed_end if committed_end is not None else last_end
        return new_text, new_last_speaker, new_last_end

    def _last_non_empty_speaker_label(self, words: list[_WordState]) -> str:
        ordered = sorted(words, key=lambda w: (w.start, w.end))
        labels = self._stabilize_speakers(ordered)
        for label in reversed(labels):
            token = str(label or '').strip()
            if token:
                return token
        return ''

    # ---- Delayed-freeze speaker re-anchoring ----------------------------------
    # When commit-hold is enabled, flushed words are buffered (re-rendered each
    # frame) instead of frozen immediately, so a late cross-window profile identity
    # can back-fill the speaker on a new turn's onset words before the marker bakes.

    def _absorb_moved_into_commit(self, moved: list[_WordState]) -> None:
        """Route flushed words either straight to the frozen committed string (legacy,
        hold disabled = byte-identical) or into the delayed-freeze pending buffer."""
        if not moved:
            return
        if self._commit_hold_seconds <= 0.0:
            self._append_to_rolling_committed_text(moved)
            return
        # Hold the flush batch as a unit; the per-batch drain re-runs the exact legacy
        # append (which dedups overlapping re-transcribed words at the committed-string
        # overlap merge), so no separate word-level dedup is needed here.
        self._pending_commit_batches.append(list(moved))

    def _pending_flat(self) -> list[_WordState]:
        return [w for batch in self._pending_commit_batches for w in batch]

    def _committed_base_text(self) -> str:
        """Committed text used as the overlay/merge base: the frozen string, plus a
        non-destructive render of any held pending words so they stay visible while
        their speaker settles."""
        if self._commit_hold_seconds <= 0.0 or not self._pending_commit_batches:
            return self._rolling_committed_text
        # Fold the held batches into a copy of the committed state the same way the
        # eventual per-batch drain will, so the previewed (and snapshotted) text is
        # exactly what gets frozen.
        text = self._rolling_committed_text
        speaker = self._rolling_committed_last_speaker
        end = self._rolling_committed_last_end
        for batch in self._pending_commit_batches:
            text, speaker, end = self._fold_words_into_committed(text, speaker, end, batch)
        return text

    def _reanchor_and_drain_pending(self, *, now: float, force_all: bool = False) -> None:
        if self._commit_hold_seconds <= 0.0:
            return
        if not self._pending_commit_batches:
            return
        self._reanchor_pending_speakers()
        self._drain_pending_commits(now=now, force_all=force_all)

    def _earliest_active_speaker(self) -> str:
        """Earliest non-empty speaker among the still-active (stable/partial) words -
        i.e. the speaker just after the pending buffer; the back-fill anchor `N`."""
        active = sorted(
            list(self._stable_words) + list(self._partial_words),
            key=lambda w: (w.start, w.end),
        )
        for w in active:
            token = str(w.speaker or '').strip()
            if token:
                return token
        return ''

    def _reanchor_pending_speakers(self) -> None:
        """Back-fill empty-speaker runs in the pending buffer. A run bounded by a known
        previous speaker P and a known next speaker N (P != N) is the profile-warmup
        lag of N's turn: attribute the whole gap to N (back-dating N's marker to the
        gap onset), or split at the local-diarization boundary when one is present."""
        pending = sorted(self._pending_flat(), key=lambda w: (w.start, w.end))
        n = len(pending)
        if n == 0:
            return
        next_known = self._earliest_active_speaker()
        last_known = str(self._rolling_committed_last_speaker or '').strip()
        i = 0
        while i < n:
            spk = str(pending[i].speaker or '').strip()
            if spk:
                last_known = spk
                i += 1
                continue
            j = i
            while j < n and not str(pending[j].speaker or '').strip():
                j += 1
            after = str(pending[j].speaker).strip() if j < n else next_known
            if after:
                self._fill_empty_run(pending, i, j, last_known, after)
                last_known = after
            i = j

    def _fill_empty_run(
        self,
        pending: list[_WordState],
        start: int,
        end: int,
        before: str,
        after: str,
    ) -> None:
        """Assign speakers to pending[start:end] (all currently empty). `after` is the
        confirmed next speaker. If `before`==`after` (or no prior) the gap is a
        continuation -> all `after`. Otherwise use the local-diarization label to find
        the true onset split; absent a clear local flip, give the whole gap to `after`
        (the warmup lag belongs to the arriving speaker, per offline validation)."""
        if start >= end or not after:
            return
        if not before or before == after:
            for k in range(start, end):
                pending[k].speaker = after
            return
        local_after = self._dominant_local_after(pending, end, after)
        local_before = self._dominant_local_before(pending, start, before)
        split = start
        if local_after and local_before and local_after != local_before:
            # first onset word whose local label has switched to the next speaker's cluster
            for k in range(start, end):
                if str(pending[k].local_speaker or '').strip() == local_after:
                    split = k
                    break
            else:
                split = start
        for k in range(start, end):
            pending[k].speaker = before if k < split else after

    @staticmethod
    def _dominant_local_after(pending: list[_WordState], end: int, after: str) -> str:
        counts: dict[str, int] = {}
        for w in pending[end:end + 6]:
            if str(w.speaker or '').strip() == after:
                loc = str(w.local_speaker or '').strip()
                if loc:
                    counts[loc] = counts.get(loc, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ''

    @staticmethod
    def _dominant_local_before(pending: list[_WordState], start: int, before: str) -> str:
        counts: dict[str, int] = {}
        for w in pending[max(0, start - 6):start]:
            if str(w.speaker or '').strip() == before:
                loc = str(w.local_speaker or '').strip()
                if loc:
                    counts[loc] = counts.get(loc, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ''

    def _drain_pending_commits(self, *, now: float, force_all: bool = False) -> None:
        """Freeze the contiguous oldest prefix of pending words that has aged past the
        commit-hold window (their speaker has had time to settle) into the frozen
        committed string. `force_all` flushes everything (end-of-stream finalize)."""
        if not self._pending_commit_batches:
            return
        ready: list[list[_WordState]] = []
        if force_all:
            ready = self._pending_commit_batches
            self._pending_commit_batches = []
        else:
            cutoff = float(now) - float(self._commit_hold_seconds)
            # Release whole batches from the front whose newest word has aged past the
            # hold window (their speaker has had time to settle). Whole-batch, in
            # order, so each drain is the exact legacy append.
            while self._pending_commit_batches:
                batch = self._pending_commit_batches[0]
                batch_end = max((float(w.end) for w in batch), default=0.0)
                if batch_end <= cutoff:
                    ready.append(self._pending_commit_batches.pop(0))
                else:
                    break
        for index, batch in enumerate(ready):
            outcome = self._apply_relabel_if_configured(batch, allow_defer=not force_all)
            if outcome is RELABEL_PENDING:
                # Round 0052 Phase B: resolution not ready -- put this batch (and everything
                # behind it, commit order is FIFO) back at the front and retry on a later drain.
                self._pending_commit_batches = ready[index:] + self._pending_commit_batches
                return
            self._append_to_rolling_committed_text(batch)

    def _apply_relabel_if_configured(self, batch: list[_WordState], *, allow_defer: bool = False):
        """Round 0048/0052: right before a batch freezes, let the injected resolver refine its
        speaker labels from a local re-diarization pass over the batch's (still-mutable) audio
        span. A str result = legacy whole-batch overwrite (round 0048); a list result = turn-aware
        per-word apply with the margin gate (round 0052); `RELABEL_PENDING` (Phase B async, only
        honoured when `allow_defer`) = resolution in flight -- returned to the caller so the drain
        can re-queue the batch. No-op on None/empty/any exception -- any failure here must never
        break the live merge path."""
        if self._relabel_resolver is None or not batch:
            return None
        try:
            start = min(float(w.start) for w in batch)
            end = max(float(w.end) for w in batch)
            resolved = self._relabel_resolver(start, end)
        except Exception:
            return None
        if resolved is RELABEL_PENDING:
            # force_all (finalize / stream end) cannot wait on a worker -- freeze with the
            # existing labels rather than blocking the pipeline.
            return RELABEL_PENDING if allow_defer else None
        if isinstance(resolved, list):
            self._apply_turn_aware_relabel(batch, resolved)
            return None
        token = str(resolved or '').strip()
        if not token:
            return None
        for w in batch:
            w.speaker = token
        return None

    def _apply_turn_aware_relabel(self, batch: list[_WordState], entries: list) -> None:
        """Round 0052: per-word relabel from labeled local-diar spans. A word inherits the entry
        containing its midpoint. The margin gate bounds the damage when the profile inventory
        itself is wrong (the Bn failure mode): a resolved profile only overwrites an existing
        non-empty label when its cosine beats the incumbent label's own cosine by
        `_relabel_margin`; empty labels are back-filled on any confident match. Words in
        unresolved gaps keep their labels."""
        spans: list[tuple[float, float, str, float, dict]] = []
        for entry in entries:
            try:
                lo = float(entry.get("start"))
                hi = float(entry.get("end"))
                token = str(entry.get("resolved") or "").strip()
                cosine = float(entry.get("resolved_cosine"))
                scores = entry.get("scores") or {}
            except Exception:
                continue
            if token and hi > lo:
                spans.append((lo, hi, token, cosine, scores if isinstance(scores, dict) else {}))
        if not spans:
            return
        margin = float(self._relabel_margin)
        for w in batch:
            midpoint = (float(w.start) + float(w.end)) / 2.0
            for (lo, hi, token, cosine, scores) in spans:
                if lo <= midpoint < hi or (midpoint == hi and lo < hi):
                    current = str(w.speaker or '').strip()
                    if not current:
                        w.speaker = token
                    elif current == token:
                        pass
                    else:
                        incumbent = scores.get(current)
                        if incumbent is None or cosine >= float(incumbent) + margin:
                            w.speaker = token
                    break

    def _normalize_output_text(self, text: str) -> str:
        if not text:
            return ''
        text = self._normalize_speaker_marker_boundaries(text)
        text = self._collapse_redundant_speaker_marker_lines(text)
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r'[ \t]+', ' ', line).strip()
            if cleaned:
                lines.append(cleaned)
            elif lines and lines[-1] != '':
                lines.append('')
        return '\n'.join(lines).strip()

    def _collapse_redundant_speaker_marker_lines(self, text: str) -> str:
        if not text:
            return ''
        out: list[str] = []
        current_marker = ''
        blank_before_line = False
        marker_pattern = re.compile(r'^\s*(>>|S\d+:|\[spk_\d+\])\s*(.*)$', flags=re.IGNORECASE)
        for raw_line in str(text).splitlines():
            line = re.sub(r'[ \t]+', ' ', raw_line).strip()
            if not line:
                blank_before_line = bool(out)
                continue
            match = marker_pattern.match(line)
            if not match:
                out.append(line)
                blank_before_line = False
                continue
            marker = match.group(1)
            marker_key = marker.lower()
            content = str(match.group(2) or '').strip()
            if marker_key == current_marker:
                if blank_before_line:
                    # Pause-separated same-speaker marker: keep the blank-line
                    # hard boundary in the output so it survives the next
                    # normalize/merge pass instead of being re-collapsed inline.
                    if out and out[-1] != '':
                        out.append('')
                    out.append(f'{marker} {content}'.strip())
                    blank_before_line = False
                    continue
                if content:
                    if out:
                        sep = '' if self._should_join_without_space(out[-1], content) else ' '
                        out[-1] = f'{out[-1].rstrip()}{sep}{content}'
                    else:
                        out.append(content)
                blank_before_line = False
                continue
            current_marker = marker_key
            out.append(f'{marker} {content}'.strip())
            blank_before_line = False
        return '\n'.join(out)

    def _merge_by_exact_overlap(self, base: str, incoming: str) -> str:
        self._last_overlap_summary = {'method': 'none', 'chars': 0, 'ratio': 0.0}
        base = self._normalize_output_text(base)
        incoming = self._normalize_output_text(incoming)
        if not base:
            self._last_overlap_summary = {'method': 'replace-empty-base', 'chars': len(incoming), 'ratio': 1.0}
            return incoming
        if not incoming:
            self._last_overlap_summary = {'method': 'keep-empty-incoming', 'chars': 0, 'ratio': 1.0}
            return base
        marker_merged = self._merge_across_leading_speaker_marker(base, incoming)
        if marker_merged:
            return marker_merged
        if self._should_use_word_overlap_merge(base, incoming):
            return self._merge_words_by_overlap(base, incoming)
        overlap = self._max_prefix_suffix_overlap(base, incoming)
        if overlap >= len(incoming):
            self._last_overlap_summary = {'method': 'exact-contained', 'chars': overlap, 'ratio': 1.0}
            return base
        if overlap > 0:
            self._last_overlap_summary = {'method': 'exact', 'chars': overlap, 'ratio': 1.0}
            return self._normalize_output_text(f'{base}{incoming[overlap:]}')
        if incoming in base[-max(16, len(incoming) * 2):]:
            self._last_overlap_summary = {'method': 'tail-contained', 'chars': len(incoming), 'ratio': 1.0}
            return base
        protected_prefix, fuzzy_base = self._split_fuzzy_overlap_base(base)
        fuzzy_overlap, fuzzy_ratio = self._max_fuzzy_prefix_suffix_overlap(fuzzy_base, incoming)
        if fuzzy_overlap > 0:
            self._last_overlap_summary = {
                'method': 'fuzzy-replace',
                'chars': fuzzy_overlap,
                'ratio': round(fuzzy_ratio, 4),
            }
            return self._normalize_output_text(f'{protected_prefix}{fuzzy_base[:-fuzzy_overlap]}{incoming}')
        short_revision = self._merge_short_cjk_revision(fuzzy_base, incoming)
        if short_revision:
            self._last_overlap_summary = short_revision['summary']
            return self._normalize_output_text(f'{protected_prefix}{short_revision["text"]}')
        interior_duplicate = self._merge_interior_duplicate_head(fuzzy_base, incoming)
        if interior_duplicate:
            self._last_overlap_summary = interior_duplicate['summary']
            return self._normalize_output_text(f'{protected_prefix}{interior_duplicate["text"]}')
        if self._starts_with_speaker_marker(incoming) and not base.endswith('\n'):
            sep = '\n'
        elif self._should_join_without_space(base, incoming):
            sep = ''
        else:
            sep = '' if base.endswith(('?', '!', ',', '.', ' ', '\n')) else ' '
        self._last_overlap_summary = {'method': 'append', 'chars': 0, 'ratio': 0.0}
        return self._normalize_output_text(f'{base}{sep}{incoming}')

    def _should_use_word_overlap_merge(self, base: str, incoming: str) -> bool:
        if self._is_cjk_source:
            return False
        return not self._contains_cjk_text(f'{base}{incoming}')

    def _merge_words_by_overlap(self, base: str, incoming: str) -> str:
        base_tokens = self._word_merge_tokens(base)
        incoming_tokens = self._word_merge_tokens(incoming)
        base_keys = self._word_overlap_keys(base_tokens)
        incoming_keys = self._word_overlap_keys(incoming_tokens)
        if not base_keys or not incoming_keys:
            sep = '\n' if self._starts_with_speaker_marker(incoming) and not base.endswith('\n') else ' '
            self._last_overlap_summary = {'method': 'word-append', 'chars': 0, 'ratio': 0.0}
            return self._normalize_output_text(f'{base}{sep}{incoming}')

        exact_words = self._max_word_prefix_suffix_overlap(base_keys, incoming_keys)
        if exact_words >= len(incoming_keys):
            self._last_overlap_summary = {
                'method': 'word-exact-contained',
                'chars': len(incoming),
                'ratio': 1.0,
                'words': exact_words,
            }
            return base
        if exact_words > 0:
            drop_to = self._token_index_after_word_count(incoming_tokens, exact_words)
            merged_tokens = base_tokens + incoming_tokens[drop_to:]
            self._last_overlap_summary = {
                'method': 'word-exact',
                'chars': 0,
                'ratio': 1.0,
                'words': exact_words,
            }
            return self._normalize_output_text(self._join_word_merge_tokens(merged_tokens))

        contained = self._incoming_words_contained_in_base_tail(base_keys, incoming_keys)
        if contained:
            self._last_overlap_summary = {
                'method': 'word-tail-contained',
                'chars': len(incoming),
                'ratio': 1.0,
                'words': len(incoming_keys),
            }
            return base

        fuzzy_words, fuzzy_ratio = self._max_fuzzy_word_prefix_suffix_overlap(base_keys, incoming_keys)
        if fuzzy_words > 0:
            keep_to = self._token_index_before_last_word_count(base_tokens, fuzzy_words)
            merged_tokens = base_tokens[:keep_to] + incoming_tokens
            merged = self._join_word_merge_tokens(merged_tokens)
            if self._word_count(base) - self._word_count(merged) > self._word_count(incoming):
                merged = self._append_words_plain(base, incoming)
                self._last_overlap_summary = {'method': 'word-append-guarded', 'chars': 0, 'ratio': 0.0}
                return merged
            self._last_overlap_summary = {
                'method': 'word-fuzzy-replace',
                'chars': 0,
                'ratio': round(fuzzy_ratio, 4),
                'words': fuzzy_words,
            }
            return self._normalize_output_text(merged)

        self._last_overlap_summary = {'method': 'word-append', 'chars': 0, 'ratio': 0.0}
        return self._append_words_plain(base, incoming)

    @classmethod
    def _word_merge_tokens(cls, text: str) -> list[str]:
        return re.findall(r'\n+|(?:>>|S\d+:|\[spk_\d+\])|[^\s]+', str(text or ''), flags=re.IGNORECASE)

    @classmethod
    def _word_overlap_keys(cls, tokens: list[str]) -> list[str]:
        keys: list[str] = []
        for token in tokens:
            key = cls._word_overlap_key(token)
            if key:
                keys.append(key)
        return keys

    @staticmethod
    def _word_overlap_key(token: str) -> str:
        value = str(token or '').strip()
        if not value or value.startswith('\n'):
            return ''
        if re.fullmatch(r'(?:>>|S\d+:|\[spk_\d+\])', value, flags=re.IGNORECASE):
            return ''
        value = re.sub(r'^[^\w]+|[^\w]+$', '', value.lower(), flags=re.UNICODE)
        return SubtitleAssembler._normalize_word(value) if value else ''

    @staticmethod
    def _max_word_prefix_suffix_overlap(base_keys: list[str], incoming_keys: list[str]) -> int:
        max_words = min(len(base_keys), len(incoming_keys))
        for size in range(max_words, 0, -1):
            if base_keys[-size:] == incoming_keys[:size]:
                return size
        return 0

    @staticmethod
    def _incoming_words_contained_in_base_tail(base_keys: list[str], incoming_keys: list[str]) -> bool:
        if not incoming_keys:
            return False
        tail = base_keys[-max(len(incoming_keys) * 2, 24):]
        width = len(incoming_keys)
        for index in range(0, len(tail) - width + 1):
            if tail[index:index + width] == incoming_keys:
                return True
        return False

    @staticmethod
    def _max_fuzzy_word_prefix_suffix_overlap(base_keys: list[str], incoming_keys: list[str]) -> tuple[int, float]:
        max_words = min(len(base_keys), len(incoming_keys), 24)
        min_words = 3
        if max_words < min_words:
            return 0, 0.0
        best_size = 0
        best_ratio = 0.0
        for size in range(max_words, min_words - 1, -1):
            left = base_keys[-size:]
            right = incoming_keys[:size]
            ratio = SequenceMatcher(None, left, right).ratio()
            if ratio >= 0.80:
                return size, ratio
            if ratio > best_ratio:
                best_size = size
                best_ratio = ratio
        if best_size >= min_words + 2 and best_ratio >= 0.74:
            return best_size, best_ratio
        return 0, 0.0

    @classmethod
    def _token_index_after_word_count(cls, tokens: list[str], word_count: int) -> int:
        seen = 0
        for index, token in enumerate(tokens):
            if cls._word_overlap_key(token):
                seen += 1
                if seen >= word_count:
                    return index + 1
        return len(tokens)

    @classmethod
    def _token_index_before_last_word_count(cls, tokens: list[str], word_count: int) -> int:
        seen = 0
        for index in range(len(tokens) - 1, -1, -1):
            if cls._word_overlap_key(tokens[index]):
                seen += 1
                if seen >= word_count:
                    return index
        return 0

    @classmethod
    def _word_count(cls, text: str) -> int:
        return len(cls._word_overlap_keys(cls._word_merge_tokens(text)))

    def _append_words_plain(self, base: str, incoming: str) -> str:
        if self._starts_with_speaker_marker(incoming) and not base.endswith('\n'):
            sep = '\n'
        elif self._should_join_without_space(base, incoming):
            sep = ''
        else:
            sep = '' if base.endswith((' ', '\n')) else ' '
        return self._normalize_output_text(f'{base}{sep}{incoming}')

    @classmethod
    def _join_word_merge_tokens(cls, tokens: list[str]) -> str:
        out = ''
        for token in tokens:
            if not token:
                continue
            if token.startswith('\n'):
                # Preserve a blank-line pause boundary (>=2 newlines, round 0008)
                # so `_collapse_redundant_speaker_marker_lines` keeps the
                # re-emitted same-speaker marker on its own line instead of
                # merging it inline. Collapsing every run to a single newline
                # flattened English pause breaks.
                if out:
                    want = '\n\n' if len(token) >= 2 else '\n'
                    trimmed = out.rstrip(' ')
                    existing = len(trimmed) - len(trimmed.rstrip('\n'))
                    needed = len(want) - existing
                    out = trimmed + ('\n' * needed if needed > 0 else '')
                continue
            if re.fullmatch(r'(?:>>|S\d+:|\[spk_\d+\])', token, flags=re.IGNORECASE):
                if out and not out.endswith('\n'):
                    out = out.rstrip() + '\n'
                out += token
                continue
            if not out or out.endswith('\n'):
                out += token
            elif cls._should_join_without_space(out, token):
                out = out.rstrip() + token
            else:
                out = out.rstrip() + ' ' + token
        return out.strip()

    def _merge_across_leading_speaker_marker(self, base: str, incoming: str) -> str:
        match = re.match(r'^\s*(?P<marker>>>|S\d+:|\[spk_\d+\])\s*(?P<content>.*)$', incoming, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return ''
        content = str(match.group('content') or '').strip()
        if not content:
            return ''
        marker_matches = list(re.finditer(r'(?:^|\n)\s*(?:>>|S\d+:|\[spk_\d+\])\s*', base, flags=re.IGNORECASE))
        if not marker_matches:
            return self._merge_leading_marker_content_against_unmarked_base(base, match.group('marker'), content)
        marker_start = marker_matches[-1].start()
        before_marker = base[:marker_start].rstrip()
        if not before_marker:
            return ''
        overlap = self._marker_prefix_suffix_overlap(before_marker, content)
        if overlap > 0:
            marker_text = match.group('marker')
            self._last_overlap_summary = {
                'method': 'marker-exact',
                'chars': int(overlap),
                'ratio': 1.0,
            }
            return self._normalize_output_text(f'{before_marker[:-overlap]}\n{marker_text} {content}')
        short_revision = self._merge_short_cjk_revision(before_marker, content)
        if short_revision:
            marker_text = match.group('marker')
            self._last_overlap_summary = {
                'method': 'marker-short-revision',
                'chars': int(short_revision['summary'].get('chars', 0) or 0),
                'ratio': float(short_revision['summary'].get('ratio', 0.0) or 0.0),
            }
            return self._normalize_output_text(f'{short_revision["text"]}\n{marker_text} {content}')
        interior_duplicate = self._merge_interior_duplicate_head(before_marker, content, prefer_incoming_head=True)
        if not interior_duplicate:
            return ''
        marker_text = match.group('marker')
        self._last_overlap_summary = {
            'method': 'marker-interior-duplicate',
            'chars': int(interior_duplicate['summary'].get('chars', 0) or 0),
            'ratio': float(interior_duplicate['summary'].get('ratio', 0.0) or 0.0),
        }
        return self._normalize_output_text(f'{interior_duplicate["text"]}\n{marker_text} {content}')

    def _merge_leading_marker_content_against_unmarked_base(self, base: str, marker_text: str, content: str) -> str:
        if not base or not content:
            return ''
        overlap = self._max_prefix_suffix_overlap(base, content)
        if overlap >= len(content):
            self._last_overlap_summary = {
                'method': 'leading-marker-exact-contained',
                'chars': int(overlap),
                'ratio': 1.0,
            }
            return self._normalize_output_text(f'{base}\n{marker_text}')
        if overlap > 0:
            self._last_overlap_summary = {
                'method': 'leading-marker-exact',
                'chars': int(overlap),
                'ratio': 1.0,
            }
            return self._normalize_output_text(f'{base}\n{marker_text} {content[overlap:]}')
        if content in base[-max(16, len(content) * 2):]:
            self._last_overlap_summary = {
                'method': 'leading-marker-tail-contained',
                'chars': len(content),
                'ratio': 1.0,
            }
            return self._normalize_output_text(f'{base}\n{marker_text}')

        protected_prefix, fuzzy_base = self._split_fuzzy_overlap_base(base)
        fuzzy_overlap, fuzzy_ratio = self._max_fuzzy_prefix_suffix_overlap(fuzzy_base, content)
        if fuzzy_overlap > 0:
            self._last_overlap_summary = {
                'method': 'leading-marker-fuzzy-replace',
                'chars': fuzzy_overlap,
                'ratio': round(fuzzy_ratio, 4),
            }
            merged_base = f'{protected_prefix}{fuzzy_base[:-fuzzy_overlap]}{content[:fuzzy_overlap]}'
            return self._normalize_output_text(f'{merged_base}\n{marker_text} {content[fuzzy_overlap:]}')

        short_revision = self._merge_short_cjk_revision(fuzzy_base, content)
        if short_revision:
            consumed = int(short_revision['summary'].get('chars', 0) or 0)
            self._last_overlap_summary = {
                'method': 'leading-marker-short-revision',
                'chars': consumed,
                'ratio': float(short_revision['summary'].get('ratio', 0.0) or 0.0),
            }
            return self._normalize_output_text(f'{protected_prefix}{short_revision["text"]}\n{marker_text} {content[consumed:]}')

        interior_duplicate = self._merge_interior_duplicate_head(fuzzy_base, content)
        if not interior_duplicate:
            return ''
        self._last_overlap_summary = {
            'method': 'leading-marker-interior-duplicate',
            'chars': int(interior_duplicate['summary'].get('chars', 0) or 0),
            'ratio': float(interior_duplicate['summary'].get('ratio', 0.0) or 0.0),
        }
        consumed = int(interior_duplicate['summary'].get('chars', 0) or 0)
        return self._normalize_output_text(f'{protected_prefix}{interior_duplicate["text"]}\n{marker_text} {content[consumed:]}')

    def _marker_prefix_suffix_overlap(self, base: str, incoming_content: str) -> int:
        max_size = min(len(base), len(incoming_content), 12)
        for size in range(max_size, 1, -1):
            left = self._compact_for_overlap(base[-size:])
            right = self._compact_for_overlap(incoming_content[:size])
            if left and left == right:
                return size
        return 0

    def _merge_short_cjk_revision(self, base: str, incoming: str) -> dict[str, object]:
        if not base or not incoming:
            return {}
        if not re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', f'{base}{incoming}'):
            return {}
        max_size = min(len(base), len(incoming), 12)
        min_size = min(5, max_size)
        if max_size < min_size:
            return {}
        best: tuple[int, float] | None = None
        for size in range(max_size, min_size - 1, -1):
            left = base[-size:]
            right = incoming[:size]
            left_key = self._compact_for_overlap(left)
            right_key = self._compact_for_overlap(right)
            if not left_key or not right_key:
                continue
            ratio = SequenceMatcher(None, left_key, right_key).ratio()
            if ratio >= 0.82:
                # Short revisions should only absorb a small continuation.
                # A long new tail is likely fresh content, not an overlap fix.
                if len(incoming[size:]) > 4:
                    continue
                best = (size, ratio)
                break
        if best is None:
            return {}
        size, ratio = best
        merged_overlap = self._short_common_supersequence(base[-size:], incoming[:size])
        return {
            'text': f'{base[:-size]}{merged_overlap}{incoming[size:]}',
            'summary': {
                'method': 'short-cjk-revision',
                'chars': int(size),
                'ratio': round(float(ratio), 4),
            },
        }

    def _merge_interior_duplicate_head(
        self,
        base: str,
        incoming: str,
        *,
        prefer_incoming_head: bool = False,
    ) -> dict[str, object]:
        if not base or not incoming:
            return {}
        if not re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', f'{base}{incoming}'):
            return {}
        lookback_chars = 32
        min_match_chars = 4
        max_match_chars = min(len(incoming), 14)
        max_remnant_chars = 6
        if max_match_chars < min_match_chars:
            return {}

        base_tail_start = max(0, len(base) - lookback_chars)
        base_tail = base[base_tail_start:]
        tail_key, tail_positions = self._compact_with_positions(base_tail)
        if not tail_key:
            return {}

        for size in range(max_match_chars, min_match_chars - 1, -1):
            incoming_head = incoming[:size]
            incoming_key = self._compact_for_overlap(incoming_head)
            if len(incoming_key) < min_match_chars:
                continue
            key_index = tail_key.find(incoming_key)
            while key_index >= 0:
                raw_start = tail_positions[key_index]
                raw_end = tail_positions[key_index + len(incoming_key) - 1] + 1
                if raw_end >= len(base_tail):
                    key_index = tail_key.find(incoming_key, key_index + 1)
                    continue
                remnant = base_tail[raw_end:]
                remnant_key = self._compact_for_overlap(remnant)
                if 0 < len(remnant_key) <= max_remnant_chars:
                    base_match_start = base_tail_start + raw_start
                    base_match_end = base_tail_start + raw_end
                    if prefer_incoming_head:
                        text = f'{base[:base_match_start].rstrip()}'
                    else:
                        text = f'{base[:base_match_end]}{incoming[size:]}'
                    return {
                        'text': text,
                        'summary': {
                            'method': 'interior-duplicate',
                            'chars': int(size),
                            'ratio': 1.0,
                        },
                    }
                key_index = tail_key.find(incoming_key, key_index + 1)
        return {}

    @staticmethod
    def _compact_with_positions(text: str) -> tuple[str, list[int]]:
        compact_chars: list[str] = []
        positions: list[int] = []
        try:
            from voice2text.stt.audio_utils import normalize_chinese_script
        except Exception:
            normalize_chinese_script = None

        for idx, char in enumerate(str(text or '')):
            lowered = char.lower()
            if lowered.isspace():
                continue
            if normalize_chinese_script is not None:
                try:
                    folded = normalize_chinese_script(lowered, 'hans')
                except Exception:
                    folded = lowered
            else:
                folded = lowered
            for folded_char in folded:
                compact_chars.append(folded_char)
                positions.append(idx)
        return ''.join(compact_chars), positions

    @staticmethod
    def _short_common_supersequence(left: str, right: str) -> str:
        out: list[str] = []
        matcher = SequenceMatcher(None, left, right)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                out.append(left[i1:i2])
            elif tag == 'delete':
                out.append(left[i1:i2])
            elif tag == 'insert':
                out.append(right[j1:j2])
            else:
                left_part = left[i1:i2]
                right_part = right[j1:j2]
                left_key = SubtitleAssembler._compact_for_overlap(left_part)
                right_key = SubtitleAssembler._compact_for_overlap(right_part)
                # Fold-aware: when one side's script-folded key is contained in the
                # other's, the longer side already carries the shorter's content, so
                # keep only the superset instead of zipping the two scripts together
                # (prevents e.g. '国家' + '個國家' -> '国個國家'). Prefer incoming
                # (right) on an exact fold tie.
                if left_key and right_key and left_key == right_key:
                    out.append(right_part)
                elif left_key and right_key and left_key in right_key:
                    out.append(right_part)
                elif left_key and right_key and right_key in left_key:
                    out.append(left_part)
                else:
                    out.append(left_part)
                    out.append(right_part)
        return ''.join(out)

    def _max_fuzzy_prefix_suffix_overlap(self, base: str, incoming: str) -> tuple[int, float]:
        """Return a near-match overlap for sliding-window STT corrections.

        WhisperX may revise a few characters in the same phrase between adjacent
        windows. When exact overlap fails, replacing the matched rolling tail
        with the newer incoming window avoids duplicate visible text and lets
        the overlay absorb those recognition corrections.
        """
        if not base or not incoming:
            return 0, 0.0
        has_cjk = bool(re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', f'{base}{incoming}'))
        min_overlap = 8 if has_cjk else 16
        max_overlap = min(len(base), len(incoming), 180)
        if max_overlap < min_overlap:
            return 0, 0.0

        best_size = 0
        best_ratio = 0.0
        threshold = 0.72 if has_cjk else 0.78
        fallback_threshold = 0.66 if has_cjk else 0.72
        for size in range(max_overlap, min_overlap - 1, -1):
            left = self._compact_for_overlap(base[-size:])
            right = self._compact_for_overlap(incoming[:size])
            if not left or not right:
                continue
            ratio = SequenceMatcher(None, left, right).ratio()
            if ratio >= threshold:
                return size, ratio
            if ratio > best_ratio:
                best_size = size
                best_ratio = ratio
        if best_size >= min_overlap * 2 and best_ratio >= fallback_threshold:
            return best_size, best_ratio
        return 0, 0.0

    @staticmethod
    def _split_fuzzy_overlap_base(base: str) -> tuple[str, str]:
        """Keep speaker markers out of fuzzy replacement ranges."""
        matches = list(re.finditer(r'(?:^|\n)\s*(?:>>|S\d+:|\[spk_\d+\])\s*', str(base or ''), flags=re.IGNORECASE))
        if not matches:
            return '', base
        split_at = matches[-1].end()
        return base[:split_at], base[split_at:]

    @staticmethod
    def _compact_for_overlap(text: str) -> str:
        compact = re.sub(r'\s+', '', str(text or '').lower())
        if not compact:
            return ''
        try:
            from voice2text.stt.audio_utils import normalize_chinese_script

            return normalize_chinese_script(compact, 'hans')
        except Exception:
            return compact

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
        normalized = re.sub(rf'([^\n])[ \t]*{marker}[ \t]*', r'\1\n\2 ', text, flags=re.IGNORECASE)
        normalized = re.sub(rf'^[ \t]*{marker}[ \t]*', r'\1 ', normalized, flags=re.IGNORECASE)
        normalized = re.sub(rf'\n[ \t]*{marker}[ \t]*', r'\n\1 ', normalized, flags=re.IGNORECASE)
        return normalized

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0
