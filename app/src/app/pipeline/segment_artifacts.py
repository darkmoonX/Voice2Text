"""Debug artifact writer for latest capture/STT segment wav snapshots."""
from __future__ import annotations

from pathlib import Path
import wave

from ..capture import AudioChunk


class SegmentArtifacts:
    def __init__(self, *, log_dir: str) -> None:
        root = Path(log_dir).resolve().parent / "segments"
        self.segment_dir = root
        self.latest_raw_segment_wav = root / "latest_segment_raw.wav"
        self.latest_stt_segment_wav = root / "latest_segment_stt.wav"

    def write_chunk(self, chunk: AudioChunk, target_path: Path) -> None:
        try:
            self.segment_dir.mkdir(parents=True, exist_ok=True)
            pcm = bytes(chunk.pcm16 or b"")
            if not pcm:
                return
            channels = max(1, int(chunk.channels))
            sample_rate = max(8000, int(chunk.sample_rate))
            with wave.open(str(target_path), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm)
        except Exception:
            # Debug artifact write failures must never interrupt runtime.
            return
