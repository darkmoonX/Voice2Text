"""Audio window sizing helpers for segment/hop buffering."""
from __future__ import annotations


def aligned_window_sizes(
    *,
    sample_rate: int,
    channels: int,
    segment_seconds: float,
    hop_seconds: float,
) -> tuple[int, int, int, int]:
    """Return aligned byte sizes:

    (bytes_per_second, frame_bytes, segment_bytes, hop_bytes)
    """
    rate = max(1, int(sample_rate))
    ch = max(1, int(channels))
    bytes_per_second = max(1, rate * ch * 2)
    frame_bytes = max(2, ch * 2)
    segment_bytes = max(int(bytes_per_second * float(segment_seconds)), bytes_per_second // 2)
    hop_bytes = max(1, int(bytes_per_second * float(hop_seconds)))
    hop_bytes = min(hop_bytes, segment_bytes)

    segment_bytes = max(frame_bytes, segment_bytes // frame_bytes * frame_bytes)
    hop_bytes = max(frame_bytes, hop_bytes // frame_bytes * frame_bytes)
    hop_bytes = min(hop_bytes, segment_bytes)
    return (bytes_per_second, frame_bytes, segment_bytes, hop_bytes)
