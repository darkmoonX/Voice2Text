"""Audio source discovery and capture backends for loopback, microphone, and app-session modes."""
from __future__ import annotations
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
try:
    import pyaudiowpatch as pyaudio
except Exception:
    pyaudio = None
if TYPE_CHECKING:
    from .config import RuntimeConfig
try:
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
except Exception:
    AudioUtilities = None
    IAudioMeterInformation = None
_SESSION_LIST_CACHE: list[str] = []
_VIRTUAL_CABLE_KEYWORDS = ('vb cable', 'vb-cable', 'vb-audio', 'virtual audio cable', 'virtual cable', 'cable output', 'cable input', 'cable-a', 'cable-b', 'voicemeeter', 'virtual audio', '虛擬')

@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_loopback: bool
    kind: str

@dataclass(frozen=True)
class LoopbackDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int

@dataclass(frozen=True)
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    channels: int

class AudioCaptureBase:
    sample_rate: int = 16000
    channels: int = 1

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def read_chunk(self, timeout: float=0.2) -> Optional[AudioChunk]:
        raise NotImplementedError

def list_audio_devices() -> list[AudioDevice]:
    """Enumerate available loopback/microphone devices for settings source selection UI."""
    if pyaudio is None:
        return []
    pa = pyaudio.PyAudio()
    devices: dict[int, AudioDevice] = {}
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_input = int(info.get('maxInputChannels', 0) or 0)
            is_loopback = bool(info.get('isLoopbackDevice', False))
            if max_input <= 0 and (not is_loopback):
                continue
            devices[idx] = AudioDevice(index=int(info['index']), name=str(info['name']), max_input_channels=max_input, default_sample_rate=int(info.get('defaultSampleRate', 0) or 0), is_loopback=is_loopback, kind='loopback' if is_loopback else 'microphone')
        try:
            for info in pa.get_loopback_device_info_generator():
                idx = int(info['index'])
                devices[idx] = AudioDevice(index=idx, name=str(info['name']), max_input_channels=int(info.get('maxInputChannels', 0) or 0), default_sample_rate=int(info.get('defaultSampleRate', 0) or 0), is_loopback=True, kind='loopback')
        except Exception:
            pass
    finally:
        pa.terminate()
    return sorted(devices.values(), key=lambda d: (d.kind != 'loopback', d.name.lower(), d.index))

def list_loopback_devices() -> list[LoopbackDevice]:
    return [LoopbackDevice(index=d.index, name=d.name, max_input_channels=d.max_input_channels, default_sample_rate=d.default_sample_rate) for d in list_audio_devices() if d.is_loopback]

def list_active_app_sessions() -> list[str]:
    """Enumerate active app-session labels used by app-gated capture mode."""
    global _SESSION_LIST_CACHE
    if AudioUtilities is not None:
        result = _collect_active_session_names()
        if result:
            _SESSION_LIST_CACHE = sorted(set(result), key=str.lower)
            return list(_SESSION_LIST_CACHE)
    return list(_SESSION_LIST_CACHE)

def _safe_get_audio_sessions() -> list[object]:
    if AudioUtilities is None:
        return []
    try:
        return list(AudioUtilities.GetAllSessions())
    except Exception:
        return []

def _collect_active_session_names() -> list[str]:
    names: set[str] = set()
    sessions = _safe_get_audio_sessions()
    for session in sessions:
        label = _format_session_label(session)
        if not label:
            continue
        names.add(label)
    return sorted(names, key=str.lower)

def _format_session_label(session: object) -> str:
    display_name = str(getattr(session, 'DisplayName', '') or '').strip()
    proc_name = _extract_process_name(session)
    if display_name.startswith('@') and 'audiosrv' in display_name.lower():
        display_name = 'System Sounds'
    if display_name and proc_name:
        if display_name.casefold() == proc_name.casefold():
            return proc_name
        return f'{display_name} ({proc_name})'
    if display_name:
        return display_name
    if proc_name:
        return proc_name
    return 'System Sounds'

def _extract_process_name(session: object) -> str:
    proc = getattr(session, 'Process', None)
    if proc is None:
        return ''
    try:
        return str(proc.name() or '').strip()
    except Exception:
        return ''

def _session_match_tokens(session: object) -> set[str]:
    tokens: set[str] = set()
    label = _format_session_label(session).strip().lower()
    if label:
        tokens.add(label)
    display_name = str(getattr(session, 'DisplayName', '') or '').strip().lower()
    if display_name:
        tokens.add(display_name)
    proc_name = _extract_process_name(session).strip().lower()
    if proc_name:
        tokens.add(proc_name)
        if proc_name.endswith('.exe'):
            tokens.add(proc_name[:-4])
    if not proc_name and (not display_name):
        tokens.add('system sounds')
    return tokens

def _token_matches(target: str, tokens: set[str]) -> bool:
    if not target.strip():
        return False
    target_variants = _target_match_variants(target)
    token_variants: set[str] = set()
    for token in tokens:
        token_variants.update(_target_match_variants(token))
    for needle in target_variants:
        for token in token_variants:
            if not needle or not token:
                continue
            if needle == token:
                return True
            if len(needle) >= 3 and needle in token:
                return True
            if len(token) >= 3 and token in needle:
                return True
    needle_canon = _canonicalize_match_text(target)
    if not needle_canon:
        return False
    for token in token_variants:
        token_canon = _canonicalize_match_text(token)
        if not token_canon:
            continue
        if needle_canon == token_canon:
            return True
        if len(needle_canon) >= 4 and needle_canon in token_canon:
            return True
        if len(token_canon) >= 4 and token_canon in needle_canon:
            return True
    return False

def _target_match_variants(value: str) -> set[str]:
    raw = (value or '').strip().lower()
    if not raw:
        return set()
    variants = {raw}
    if raw.endswith('.exe'):
        variants.add(raw[:-4])
    for inner in re.findall('\\(([^)]+)\\)', raw):
        token = inner.strip().lower()
        if token:
            variants.add(token)
            if token.endswith('.exe'):
                variants.add(token[:-4])
    parts = [part.strip() for part in re.split('[|,;/]', raw) if part.strip()]
    variants.update(parts)
    canonical = _canonicalize_match_text(raw)
    if canonical:
        variants.add(canonical)
    return {item for item in variants if item}

def _canonicalize_match_text(value: str) -> str:
    lowered = (value or '').strip().lower()
    if not lowered:
        return ''
    lowered = lowered.replace('.exe', '')
    lowered = re.sub('[^0-9a-z\\u4e00-\\u9fff]+', '', lowered)
    return lowered

class _SingleDeviceAudioCapture(AudioCaptureBase):

    def __init__(self, source_kind: str, device_index: Optional[int]=None, frames_per_buffer: int=2048, preferred_sample_rate: Optional[int]=None, on_error: Optional[Callable[[str], None]]=None) -> None:
        self._source_kind = source_kind
        self._device_index = device_index
        self._frames_per_buffer = frames_per_buffer
        self._preferred_sample_rate = preferred_sample_rate
        self._on_error = on_error
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=512)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sample_rate = 16000
        self.channels = 1

    def start(self) -> None:
        if pyaudio is None:
            raise RuntimeError('pyaudiowpatch is not installed. Install it to enable audio capture.')
        if self._running.is_set():
            return
        self._pa = pyaudio.PyAudio()
        info = self._resolve_device_info()
        self.sample_rate = int(self._preferred_sample_rate or info.get('defaultSampleRate', 16000) or 16000)
        max_input = int(info.get('maxInputChannels', 1) or 1)
        self.channels = int(max(1, min(2, max_input)))
        self._stream = self._open_stream_with_fallback(info)
        self._running.set()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read_chunk(self, timeout: float=0.2) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _reader_loop(self) -> None:
        while self._running.is_set() and self._stream is not None:
            try:
                data = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
                if not data:
                    continue
                if len(data) % 2 != 0:
                    data = data[:-1]
                if not data:
                    continue
                chunk = AudioChunk(data, self.sample_rate, self.channels)
                try:
                    self._queue.put(chunk, timeout=0.1)
                except queue.Full:
                    pass
            except Exception as exc:
                if not self._running.is_set():
                    break
                self._emit_error(f'Audio read failed: {exc}')
                break
        self._running.clear()

    def _resolve_device_info(self) -> dict:
        assert self._pa is not None
        if self._device_index is not None:
            info = self._pa.get_device_info_by_index(self._device_index)
            if self._source_kind == 'loopback':
                if info.get('isLoopbackDevice', False):
                    return info
                match = self._find_loopback_by_output_name(str(info.get('name', '')))
                if match is not None:
                    return match
                return info
            if int(info.get('maxInputChannels', 0) or 0) > 0:
                return info
            raise RuntimeError('Selected input device does not support microphone capture.')
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception as exc:
            raise RuntimeError('WASAPI host is not available.') from exc
        if self._source_kind == 'loopback':
            default_output = self._pa.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
            if default_output.get('isLoopbackDevice', False):
                return default_output
            match = self._find_loopback_by_output_name(str(default_output.get('name', '')))
            if match is not None:
                return match
            raise RuntimeError('No loopback device found for default output endpoint.')
        default_input_idx = int(wasapi_info.get('defaultInputDevice', -1))
        if default_input_idx >= 0:
            return self._pa.get_device_info_by_index(default_input_idx)
        try:
            return self._pa.get_default_input_device_info()
        except Exception as exc:
            raise RuntimeError('No default microphone input device found.') from exc

    def _find_loopback_by_output_name(self, output_name: str) -> Optional[dict]:
        assert self._pa is not None
        output_name = output_name.lower().strip()
        try:
            for dev in self._pa.get_loopback_device_info_generator():
                loop_name = str(dev.get('name', '')).lower()
                if output_name and output_name in loop_name:
                    return dev
        except Exception:
            return None
        return None

    def _open_stream_with_fallback(self, device_info: dict):
        assert self._pa is not None
        device_index = int(device_info['index'])
        try:
            return self._pa.open(format=pyaudio.paInt16, channels=self.channels, rate=self.sample_rate, input=True, frames_per_buffer=self._frames_per_buffer, input_device_index=device_index)
        except Exception:
            self.sample_rate = int(device_info.get('defaultSampleRate', 16000) or 16000)
            max_input = int(device_info.get('maxInputChannels', 1) or 1)
            self.channels = int(max(1, min(2, max_input)))
            return self._pa.open(format=pyaudio.paInt16, channels=self.channels, rate=self.sample_rate, input=True, frames_per_buffer=self._frames_per_buffer, input_device_index=device_index)

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)

class LoopbackAudioCapture(_SingleDeviceAudioCapture):

    def __init__(self, device_index: Optional[int]=None, frames_per_buffer: int=2048, preferred_sample_rate: Optional[int]=None, on_error: Optional[Callable[[str], None]]=None) -> None:
        super().__init__(source_kind='loopback', device_index=device_index, frames_per_buffer=frames_per_buffer, preferred_sample_rate=preferred_sample_rate, on_error=on_error)

class MicrophoneAudioCapture(_SingleDeviceAudioCapture):

    def __init__(self, device_index: Optional[int]=None, frames_per_buffer: int=2048, preferred_sample_rate: Optional[int]=None, on_error: Optional[Callable[[str], None]]=None) -> None:
        super().__init__(source_kind='microphone', device_index=device_index, frames_per_buffer=frames_per_buffer, preferred_sample_rate=preferred_sample_rate, on_error=on_error)

class AppSessionCapture(LoopbackAudioCapture):

    def __init__(self, app_names: list[str] | None=None, device_index: Optional[int]=None, frames_per_buffer: int=2048, preferred_sample_rate: Optional[int]=None, on_error: Optional[Callable[[str], None]]=None, on_status: Optional[Callable[[str], None]]=None) -> None:
        super().__init__(device_index=device_index, frames_per_buffer=frames_per_buffer, preferred_sample_rate=preferred_sample_rate, on_error=on_error)
        names = app_names or []
        self._app_names = [name.strip().lower() for name in names if name.strip()]
        self._on_status = on_status
        self._warned_no_session_api = False
        self._selected_peak_floor = 0.0025
        self._selected_peak_strict = 0.0055
        self._dominance_ratio = 1.35
        self._dominance_margin = 0.0018
        self._gate_hold_seconds = 0.9
        self._active_gate_until = 0.0
        self._last_nomatch_notice_at = 0.0
        self._debug_state: dict[str, object] = {'decision': 'init', 'selected_peak': 0.0, 'other_peak': 0.0, 'ratio': 0.0, 'matched_sessions': [], 'app_targets': list(self._app_names)}

    def start(self) -> None:
        super().start()
        if self._on_status is not None:
            if self._app_names:
                self._on_status('App source mode enabled. Uses session-dominance gating on loopback capture.')
            else:
                self._on_status('App source mode enabled without app name; using full loopback stream.')

    def read_chunk(self, timeout: float=0.2) -> Optional[AudioChunk]:
        chunk = super().read_chunk(timeout=timeout)
        if chunk is None:
            return None
        if not self._app_names:
            return chunk
        if self._is_target_audio_dominant():
            return chunk
        return None

    def _is_target_audio_dominant(self) -> bool:
        now = time.monotonic()
        if AudioUtilities is None or IAudioMeterInformation is None:
            if not self._warned_no_session_api and self._on_status is not None:
                self._on_status('pycaw not installed; app mode falls back to full loopback capture.')
                self._warned_no_session_api = True
            self._set_debug_state(decision='pycaw-unavailable', selected_peak=0.0, other_peak=0.0, matched_sessions=[], passed=True)
            return True
        sessions = _safe_get_audio_sessions()
        if not sessions:
            self._set_debug_state(decision='no-sessions', selected_peak=0.0, other_peak=0.0, matched_sessions=[], passed=False)
            return False
        selected_peak = 0.0
        other_peak = 0.0
        matched_labels: list[str] = []
        active_labels: list[str] = []
        for session in sessions:
            tokens = _session_match_tokens(session)
            if not tokens:
                continue
            label = _format_session_label(session)
            if label and len(active_labels) < 8:
                active_labels.append(label)
            peak = _session_peak_value(session)
            if peak <= 0.0001:
                continue
            if any((_token_matches(target, tokens) for target in self._app_names)):
                selected_peak = max(selected_peak, peak)
                if label and len(matched_labels) < 6 and (label not in matched_labels):
                    matched_labels.append(label)
            else:
                other_peak = max(other_peak, peak)
        if not matched_labels:
            if self._on_status is not None and now - self._last_nomatch_notice_at >= 12.0:
                targets = ', '.join(self._app_names) if self._app_names else '(none)'
                active = ', '.join(active_labels) if active_labels else '(none)'
                self._on_status(f'App session match not found for current targets. targets={targets}; active_sessions={active}')
                self._last_nomatch_notice_at = now
            self._set_debug_state(decision='no-target-session-match', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=[], passed=False)
            return False
        if selected_peak < self._selected_peak_floor:
            if now < self._active_gate_until and selected_peak >= self._selected_peak_floor * 0.6:
                self._set_debug_state(decision='hold-window', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=True)
                return True
            self._set_debug_state(decision='target-below-floor', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=False)
            return False
        if other_peak <= 0.0008:
            self._active_gate_until = now + self._gate_hold_seconds
            self._set_debug_state(decision='target-active-no-other', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=True)
            return True
        if selected_peak >= other_peak * self._dominance_ratio:
            self._active_gate_until = now + self._gate_hold_seconds
            self._set_debug_state(decision='target-dominant-ratio', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=True)
            return True
        if selected_peak >= self._selected_peak_strict and selected_peak - other_peak >= self._dominance_margin:
            self._active_gate_until = now + self._gate_hold_seconds
            self._set_debug_state(decision='target-dominant-margin', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=True)
            return True
        if now < self._active_gate_until and selected_peak >= self._selected_peak_floor * 0.6:
            self._set_debug_state(decision='hold-window', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=True)
            return True
        self._set_debug_state(decision='target-not-strong-enough', selected_peak=selected_peak, other_peak=other_peak, matched_sessions=matched_labels, passed=False)
        return False

    def _set_debug_state(self, *, decision: str, selected_peak: float, other_peak: float, matched_sessions: list[str], passed: bool) -> None:
        ratio = selected_peak / max(other_peak, 1e-06) if other_peak > 0.0 else float('inf')
        self._debug_state = {'decision': decision, 'selected_peak': float(selected_peak), 'other_peak': float(other_peak), 'ratio': float(ratio), 'matched_sessions': list(matched_sessions), 'app_targets': list(self._app_names), 'passed': bool(passed), 'timestamp': time.time()}

    def get_debug_state(self) -> dict[str, object]:
        return dict(self._debug_state)

def _session_peak_value(session: object) -> float:
    if IAudioMeterInformation is None:
        return 0.0
    try:
        meter = session._ctl.QueryInterface(IAudioMeterInformation)
        return float(meter.GetPeakValue())
    except Exception:
        return 0.0

class MixedAudioCapture(AudioCaptureBase):

    def __init__(self, captures: list[AudioCaptureBase], weights: Optional[list[float]]=None, target_sample_rate: int=16000, channel_mode: str='mono', on_error: Optional[Callable[[str], None]]=None) -> None:
        if not captures:
            raise ValueError('MixedAudioCapture requires at least one source capture.')
        self._captures = captures
        self._weights = weights or [1.0] * len(captures)
        self._target_sample_rate = max(8000, int(target_sample_rate))
        self._channel_mode = channel_mode
        self._on_error = on_error
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=512)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sample_rate = self._target_sample_rate
        self.channels = 1

    def start(self) -> None:
        if self._running.is_set():
            return
        for capture in self._captures:
            capture.start()
        self._running.set()
        self._thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        for capture in self._captures:
            capture.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read_chunk(self, timeout: float=0.2) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _mix_loop(self) -> None:
        while self._running.is_set():
            active_tracks: list[tuple[float, np.ndarray]] = []
            for (idx, capture) in enumerate(self._captures):
                chunk = capture.read_chunk(timeout=0.03)
                if chunk is None:
                    continue
                weight = self._weights[idx] if idx < len(self._weights) else 1.0
                if weight == 0.0:
                    continue
                audio = _pcm16_to_mono_float(chunk.pcm16, chunk.channels, self._channel_mode)
                if audio.size == 0:
                    continue
                resampled = _resample(audio, chunk.sample_rate, self.sample_rate)
                if resampled.size == 0:
                    continue
                active_tracks.append((weight, resampled))
            if not active_tracks:
                time.sleep(0.005)
                continue
            try:
                mixed = _mix_tracks(active_tracks)
                pcm16 = (mixed * 32767.0).astype(np.int16).tobytes()
                out_chunk = AudioChunk(pcm16, self.sample_rate, 1)
                self._queue.put(out_chunk, timeout=0.1)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(f'Audio mix failed: {exc}')

def build_capture_from_config(config: RuntimeConfig, on_error: Optional[Callable[[str], None]]=None, on_status: Optional[Callable[[str], None]]=None) -> AudioCaptureBase:
    """Create capture backend instance from RuntimeConfig source_mode/source selection settings."""
    mode = (config.source_mode or 'loopback').strip().lower()
    source_indices = list(config.source_device_indices)
    devices = list_audio_devices()
    by_index = {d.index: d for d in devices}
    valid_indices: list[int] = []
    for idx in source_indices:
        dev = by_index.get(idx)
        if dev is None:
            if on_status is not None:
                on_status(f'Source index {idx} no longer exists; skipping.')
            continue
        valid_indices.append(idx)
    source_indices = valid_indices
    if config.device_index is not None and (not source_indices):
        if config.device_index in by_index:
            source_indices = [config.device_index]
        elif on_status is not None:
            on_status(f'Configured device index {config.device_index} no longer exists; using default source.')
    primary_index = source_indices[0] if source_indices else None
    if mode == 'microphone':
        return MicrophoneAudioCapture(device_index=primary_index, on_error=on_error)
    if mode == 'loopback' and len(source_indices) > 1:
        captures: list[AudioCaptureBase] = [LoopbackAudioCapture(device_index=idx, on_error=on_error) for idx in source_indices]
        return MixedAudioCapture(captures=captures, weights=None, channel_mode=config.source_channel_mode, on_error=on_error)
    if mode == 'mix':
        if not source_indices:
            raise RuntimeError('Mix source mode requires --source-devices indices.')
        captures: list[AudioCaptureBase] = []
        for idx in source_indices:
            dev = by_index.get(idx)
            if dev is not None and dev.is_loopback:
                captures.append(LoopbackAudioCapture(device_index=idx, on_error=on_error))
            else:
                captures.append(MicrophoneAudioCapture(device_index=idx, on_error=on_error))
        weights = list(config.source_mix_weights)
        if weights and len(weights) != len(captures):
            if on_status is not None:
                on_status('Mix weights count mismatch; falling back to equal weights.')
            weights = []
        return MixedAudioCapture(captures=captures, weights=weights or None, channel_mode=config.source_channel_mode, on_error=on_error)
    if mode == 'app':
        app_names = [name for name in config.source_app_names if name.strip()]
        if not app_names and config.source_app_name.strip():
            app_names = [config.source_app_name.strip()]
        virtual_index = None
        if source_indices:
            selected = by_index.get(source_indices[0])
            if selected is not None and selected.is_loopback and _is_virtual_cable_name(selected.name):
                virtual_index = source_indices[0]
            elif selected is not None and on_status is not None:
                selected_name = selected.name if selected is not None else str(source_indices[0])
                on_status(f'Selected app-mode source is not a virtual cable loopback endpoint; using session-gated app capture. selected={selected_name}')
        elif app_names:
            auto_virtual = _find_virtual_cable_loopback_device(devices)
            if auto_virtual is not None:
                virtual_index = auto_virtual.index
                if on_status is not None:
                    on_status(f'App mode auto-selected virtual cable endpoint for stricter isolation: {auto_virtual.name}')
        if virtual_index is not None:
            if on_status is not None:
                selected = ', '.join(app_names) if app_names else '(not specified)'
                on_status(f'App mode uses virtual audio line loopback for strict isolation. Route target app(s) to this device in Windows volume mixer. targets={selected}')
            return LoopbackAudioCapture(device_index=virtual_index, on_error=on_error)
        if on_status is not None:
            has_virtual_cable = _find_virtual_cable_loopback_device(devices) is not None
            if has_virtual_cable:
                on_status('App mode uses session-gated capture on current loopback endpoint. VB-CABLE isolation is optional. If CABLE meter has no activity, keep --source-devices empty and continue with session-gated capture.')
            else:
                on_status('App mode uses session-gated capture on current loopback endpoint. No VB-CABLE loopback endpoint detected; session-gated capture remains active.')
        return AppSessionCapture(app_names=app_names, device_index=primary_index, on_error=on_error, on_status=on_status)
    if mode != 'loopback' and on_status is not None:
        on_status(f"Unknown source mode '{mode}', fallback to loopback.")
    return LoopbackAudioCapture(device_index=primary_index, on_error=on_error)

def _pcm16_to_mono_float(pcm16: bytes, channels: int, channel_mode: str='mono') -> np.ndarray:
    if not pcm16:
        return np.zeros((0,), dtype=np.float32)
    audio = np.frombuffer(pcm16, dtype=np.int16)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if channels > 1:
        usable = audio.size // channels * channels
        if usable <= 0:
            return np.zeros((0,), dtype=np.float32)
        matrix = audio[:usable].reshape(-1, channels).astype(np.float32)
        mode = channel_mode.lower().strip()
        if mode == 'left':
            mixed = matrix[:, 0]
        elif mode == 'right':
            mixed = matrix[:, min(1, channels - 1)]
        else:
            mixed = matrix.mean(axis=1)
    else:
        mixed = audio.astype(np.float32)
    return (mixed / 32768.0).astype(np.float32)

def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    target_size = int(audio.size * dst_rate / max(1, src_rate))
    if target_size <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, audio.size - 1, num=target_size, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)

def _mix_tracks(tracks: list[tuple[float, np.ndarray]]) -> np.ndarray:
    max_len = max((track.size for (_, track) in tracks))
    mix = np.zeros((max_len,), dtype=np.float32)
    total_weight = 0.0
    for (weight, track) in tracks:
        if track.size < max_len:
            padded = np.pad(track, (0, max_len - track.size), mode='constant')
        else:
            padded = track
        mix += padded * float(weight)
        total_weight += abs(float(weight))
    if total_weight <= 0.0:
        total_weight = float(len(tracks))
    return np.clip(mix / total_weight, -1.0, 1.0)

def _find_virtual_cable_loopback_device(devices: list[AudioDevice]) -> Optional[AudioDevice]:
    for dev in devices:
        if not dev.is_loopback:
            continue
        if _is_virtual_cable_name(dev.name):
            return dev
    return None

def _is_virtual_cable_name(name: str) -> bool:
    lowered = (name or '').lower()
    return any((token in lowered for token in _VIRTUAL_CABLE_KEYWORDS))
