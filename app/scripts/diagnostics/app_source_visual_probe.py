"""Visual probe script for app-source mode behavior during manual debugging."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk, _pcm16_to_mono_float, build_capture_from_config, list_active_app_sessions
from voice2text.config import RuntimeConfig


def _parse_str_csv(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _bar(value: float, max_value: float, width: int = 24) -> str:
    safe_max = max(1.0e-6, float(max_value))
    ratio = max(0.0, min(1.0, float(value) / safe_max))
    fill = int(round(ratio * width))
    fill = max(0, min(width, fill))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual probe for app-mode capture. Shows chunk/segment meters and app gate decisions."
    )
    parser.add_argument("--app-names", default="", help="Comma-separated target app names, e.g. msedge.exe,discord.exe")
    parser.add_argument("--source-devices", default="", help="Optional comma-separated source indices")
    parser.add_argument("--seconds", type=float, default=30.0, help="Probe duration")
    parser.add_argument("--segment-seconds", type=float, default=6.0, help="Segment length")
    parser.add_argument("--hop-seconds", type=float, default=1.5, help="Hop length")
    parser.add_argument("--rms-threshold", type=float, default=0.008, help="Segment RMS threshold")
    parser.add_argument("--show-sessions", action="store_true", help="Print current mixer app sessions before probing")
    args = parser.parse_args()

    app_names = _parse_str_csv(args.app_names)
    source_indices = _parse_int_csv(args.source_devices)

    if args.show_sessions:
        sessions = list_active_app_sessions()
        print("Mixer sessions:")
        if sessions:
            for item in sessions:
                print(f"- {item}")
        else:
            print("- (none)")
        print()

    cfg = RuntimeConfig(
        source_mode="app",
        source_device_indices=source_indices,
        source_app_name=app_names[0] if app_names else "",
        source_app_names=app_names,
        segment_seconds=max(0.5, float(args.segment_seconds)),
        hop_seconds=max(0.1, float(args.hop_seconds)),
        source_channel_mode="mono",
    )

    def on_status(msg: str) -> None:
        print(f"[status] {msg}")

    def on_error(msg: str) -> None:
        print(f"[error] {msg}")

    capture = build_capture_from_config(cfg, on_error=on_error, on_status=on_status)
    capture.start()

    print("\n=== App Capture Visual Probe ===")
    print(f"targets={app_names if app_names else ['(all)']}")
    print(f"source_indices={source_indices}")

    stream_rate = int(getattr(capture, "sample_rate", 16000) or 16000)
    stream_channels = int(getattr(capture, "channels", 1) or 1)

    def aligned_sizes(rate: int, channels: int) -> tuple[int, int, int, int]:
        bytes_per_second = max(1, rate * channels * 2)
        frame_bytes = max(2, channels * 2)
        segment_bytes = max(int(bytes_per_second * cfg.segment_seconds), bytes_per_second // 2)
        hop_bytes = max(1, int(bytes_per_second * cfg.hop_seconds))
        hop_bytes = min(hop_bytes, segment_bytes)

        segment_bytes = max(frame_bytes, (segment_bytes // frame_bytes) * frame_bytes)
        hop_bytes = max(frame_bytes, (hop_bytes // frame_bytes) * frame_bytes)
        hop_bytes = min(hop_bytes, segment_bytes)

        return bytes_per_second, frame_bytes, segment_bytes, hop_bytes

    _, frame_bytes, segment_bytes, hop_bytes = aligned_sizes(stream_rate, stream_channels)

    buffer = bytearray()
    deadline = time.monotonic() + max(3.0, float(args.seconds))
    last_no_chunk_log = 0.0
    segment_count = 0

    try:
        while time.monotonic() < deadline:
            chunk = capture.read_chunk(timeout=0.25)
            now = time.monotonic()

            if chunk is None:
                if (now - last_no_chunk_log) >= 1.0:
                    print("[probe] no chunk received in last 1s")
                    last_no_chunk_log = now
                continue

            stream_rate = int(chunk.sample_rate)
            stream_channels = int(chunk.channels)
            if stream_rate <= 0 or stream_channels <= 0:
                continue

            if len(chunk.pcm16) < frame_bytes:
                continue

            audio = _pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode="mono")
            chunk_rms = _rms(audio)
            chunk_peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

            buffer.extend(chunk.pcm16)
            if len(buffer) > segment_bytes * 5:
                del buffer[: len(buffer) - (segment_bytes * 5)]

            while len(buffer) >= segment_bytes:
                segment_count += 1
                window = bytes(buffer[:segment_bytes])
                del buffer[:hop_bytes]

                segment_chunk = AudioChunk(
                    pcm16=window,
                    sample_rate=stream_rate,
                    channels=stream_channels,
                )
                seg_audio = _pcm16_to_mono_float(
                    segment_chunk.pcm16,
                    segment_chunk.channels,
                    channel_mode="mono",
                )
                seg_rms = _rms(seg_audio)
                seg_peak = float(np.max(np.abs(seg_audio))) if seg_audio.size > 0 else 0.0
                seg_gate = seg_rms >= float(args.rms_threshold)

                debug_state: dict[str, object] = {}
                if hasattr(capture, "get_debug_state"):
                    try:
                        debug_state = getattr(capture, "get_debug_state")()
                    except Exception:
                        debug_state = {}

                app_gate = bool(debug_state.get("passed", True))
                selected_peak = float(debug_state.get("selected_peak", 0.0) or 0.0)
                other_peak = float(debug_state.get("other_peak", 0.0) or 0.0)
                ratio = float(debug_state.get("ratio", 0.0) or 0.0)
                decision = str(debug_state.get("decision", "n/a") or "n/a")
                matched = debug_state.get("matched_sessions", [])
                if isinstance(matched, list):
                    matched_text = "; ".join(str(item) for item in matched[:2])
                else:
                    matched_text = ""

                elapsed = max(0.0, float(args.seconds) - max(0.0, deadline - now))
                print(
                    f"[{elapsed:6.2f}s][seg#{segment_count:03d}] "
                    f"chunk_rms={chunk_rms:0.4f} {_bar(chunk_rms, 0.05)} "
                    f"chunk_peak={chunk_peak:0.4f} "
                    f"seg_rms={seg_rms:0.4f} {_bar(seg_rms, 0.05)} "
                    f"seg_peak={seg_peak:0.4f} "
                    f"seg_gate={'PASS' if seg_gate else 'SKIP'} "
                    f"app_gate={'PASS' if app_gate else 'DROP'} "
                    f"sel={selected_peak:0.4f} oth={other_peak:0.4f} ratio={ratio:0.2f} "
                    f"decision={decision} matched={matched_text}"
                )

    finally:
        capture.stop()

    print("\n[done] app source visual probe finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
