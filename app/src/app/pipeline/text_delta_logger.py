"""Incremental text delta logging adapter."""
from __future__ import annotations

import re
from typing import Callable


class TextDeltaLogger:
    def __init__(self, emit: Callable[[str, str], None], max_entry_chars: int = 180) -> None:
        self._emit = emit
        self._max_entry_chars = max(48, int(max_entry_chars))
        self._last_source = ''
        self._last_translated = ''

    def reset(self) -> None:
        self._last_source = ''
        self._last_translated = ''

    def log(self, prefix: str, text: str, *, translated: bool) -> None:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return
        previous = self._last_translated if translated else self._last_source
        delta = self._extract_incremental_delta(previous, cleaned)
        if not delta:
            return
        for part in self._split_log_chunks(delta, self._max_entry_chars):
            self._emit(prefix, part)
        tail = cleaned[-4000:]
        if translated:
            self._last_translated = tail
        else:
            self._last_source = tail

    def _extract_incremental_delta(self, previous: str, current: str) -> str:
        prev = re.sub(r'\s+', ' ', previous).strip()
        curr = re.sub(r'\s+', ' ', current).strip()
        if not curr:
            return ''
        if not prev:
            return curr
        if curr.startswith(prev):
            return curr[len(prev):].strip()
        if curr in prev:
            return ''
        overlap = self._max_prefix_suffix_overlap(prev, curr)
        if overlap > 0:
            return curr[overlap:].strip()
        return curr

    @staticmethod
    def _split_log_chunks(text: str, max_chars: int) -> list[str]:
        chunks: list[str] = []
        remaining = re.sub(r'\s+', ' ', text).strip()
        if not remaining:
            return chunks
        limit = max(48, int(max_chars))
        while len(remaining) > limit:
            cut = remaining.rfind(' ', 0, limit)
            if cut < limit // 2:
                cut = limit
            part = remaining[:cut].strip()
            if part:
                chunks.append(part)
            remaining = remaining[cut:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0
