"""Audio preprocessing stages (WebRTC/RNNoise/spectral/gain) and pipeline composition."""
from __future__ import annotations
from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Protocol
import numpy as np
from ..audio_capture import AudioChunk
from .audio_utils import pcm16_to_mono_float, resample, to_int16_pcm
if TYPE_CHECKING:
    from ..config import RuntimeConfig

@dataclass(frozen=True)
class PreprocessStepResult:
    name: str
    active: bool
    reason: str
    metrics: dict[str, float]

class AudioPreprocessStage(Protocol):
    name: str

    @property
    def active(self) -> bool:
        ...

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, PreprocessStepResult]:
        ...

class _BaseStage:
    name = 'base'

    @property
    def active(self) -> bool:
        return True

    def _result(self, *, active: bool, reason: str, metrics: dict[str, float] | None=None) -> PreprocessStepResult:
        return PreprocessStepResult(name=self.name, active=active, reason=reason, metrics=metrics or {})

class WebRTCAudioProcessingStage(_BaseStage):

    def __init__(self, *, name: str, enable_ns: bool=False, enable_agc: bool=False, enable_aec: bool=False, sample_rate: int=16000) -> None:
        self.name = name
        self._enable_ns = bool(enable_ns)
        self._enable_agc = bool(enable_agc)
        self._enable_aec = bool(enable_aec)
        self._sample_rate = int(sample_rate)
        self._processor = self._create_processor()
        self._unavailable_reason = '' if self._processor is not None else 'module-unavailable'

    @property
    def active(self) -> bool:
        return self._processor is not None

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, PreprocessStepResult]:
        if audio.size == 0:
            return (audio, self._result(active=self.active, reason='empty-audio'))
        if self._processor is None:
            return (audio, self._result(active=False, reason=self._unavailable_reason))
        pcm = to_int16_pcm(audio)
        try:
            processed = self._call_processor(pcm, sample_rate)
        except Exception:
            return (audio, self._result(active=False, reason='process-error'))
        if not processed:
            return (audio, self._result(active=False, reason='empty-output'))
        out = pcm16_to_mono_float(processed, 1, channel_mode='mono')
        if out.size == 0:
            return (audio, self._result(active=False, reason='decode-empty'))
        if out.size != audio.size:
            out = resample(out, sample_rate, sample_rate)
            if out.size > audio.size:
                out = out[:audio.size]
            elif out.size < audio.size:
                out = np.pad(out, (0, audio.size - out.size), mode='constant')
        return (out.astype(np.float32), self._result(active=True, reason='processed', metrics={'rms_in': _rms(audio), 'rms_out': _rms(out)}))

    def _create_processor(self) -> object | None:
        try:
            module = importlib.import_module('webrtc_audio_processing')
        except Exception:
            return None
        cls = getattr(module, 'AudioProcessingModule', None)
        if cls is None:
            cls = getattr(module, 'AudioProcessing', None)
        if cls is None:
            return None
        init_attempts = ({'enable_ns': self._enable_ns, 'enable_agc': self._enable_agc, 'enable_aec': self._enable_aec}, {'ns': self._enable_ns, 'agc': self._enable_agc, 'aec': self._enable_aec}, {})
        processor = None
        for kwargs in init_attempts:
            try:
                processor = cls(**kwargs)
                break
            except Exception:
                continue
        if processor is None:
            return None
        for (method_name, args) in (('set_stream_format', (self._sample_rate, 1)), ('set_sample_rate', (self._sample_rate,))):
            method = getattr(processor, method_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception:
                pass
        self._set_feature(processor, ('set_ns', 'set_noise_suppression'), self._enable_ns)
        self._set_feature(processor, ('set_agc', 'set_gain_control'), self._enable_agc)
        self._set_feature(processor, ('set_aec', 'set_echo_cancellation'), self._enable_aec)
        return processor

    @staticmethod
    def _set_feature(processor: object, names: tuple[str, ...], enabled: bool) -> None:
        for name in names:
            method = getattr(processor, name, None)
            if method is None:
                continue
            try:
                method(bool(enabled))
                return
            except Exception:
                continue

    def _call_processor(self, pcm: bytes, sample_rate: int) -> bytes:
        processor = self._processor
        assert processor is not None
        for method_name in ('process_stream', 'process', 'process_capture_stream'):
            method = getattr(processor, method_name, None)
            if method is None:
                continue
            try:
                result = method(pcm)
            except TypeError:
                result = method(pcm, sample_rate, 1)
            if isinstance(result, bytes):
                return result
            if isinstance(result, bytearray):
                return bytes(result)
            if isinstance(result, np.ndarray):
                return to_int16_pcm(result.astype(np.float32))
        raise RuntimeError('unsupported-webrtc-api')

class RNNoiseStage(_BaseStage):
    name = 'rnnoise'

    def __init__(self, sample_rate: int=16000) -> None:
        self._sample_rate = int(sample_rate)
        self._processor = self._create_processor()

    @property
    def active(self) -> bool:
        return self._processor is not None

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, PreprocessStepResult]:
        if audio.size == 0:
            return (audio, self._result(active=self.active, reason='empty-audio'))
        if self._processor is None:
            return (audio, self._result(active=False, reason='module-unavailable'))
        frame_size = max(1, int(sample_rate * 0.03))
        if audio.size < frame_size:
            return (audio, self._result(active=True, reason='short-window'))
        out = np.array(audio, dtype=np.float32, copy=True)
        try:
            for start in range(0, out.size - frame_size + 1, frame_size):
                out[start:start + frame_size] = self._process_frame(out[start:start + frame_size])
        except Exception:
            return (audio, self._result(active=False, reason='process-error'))
        return (out, self._result(active=True, reason='processed', metrics={'rms_in': _rms(audio), 'rms_out': _rms(out)}))

    @staticmethod
    def _create_processor() -> object | None:
        for module_name in ('rnnoise', 'pyrnnoise'):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for cls_name in ('RNNoise', 'RNNoiseDenoiser', 'Denoiser'):
                cls = getattr(module, cls_name, None)
                if cls is None:
                    continue
                try:
                    return cls()
                except Exception:
                    continue
        return None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        processor = self._processor
        assert processor is not None
        for method_name in ('process_frame', 'process', 'denoise'):
            method = getattr(processor, method_name, None)
            if method is None:
                continue
            try:
                result = method(frame)
            except TypeError:
                result = method(to_int16_pcm(frame))
            if isinstance(result, bytes):
                return pcm16_to_mono_float(result, 1, channel_mode='mono')[:frame.size]
            arr = np.asarray(result, dtype=np.float32)
            if arr.size:
                if arr.size != frame.size:
                    arr = np.resize(arr, frame.size)
                return arr
        raise RuntimeError('unsupported-rnnoise-api')

class SpectralGateStage(_BaseStage):
    name = 'spectral-gate'

    def __init__(self, *, noise_update: float=0.08, reduction: float=0.72, speech_rms: float=0.018) -> None:
        self._noise_update = max(0.001, min(1.0, float(noise_update)))
        self._reduction = max(0.0, min(0.95, float(reduction)))
        self._speech_rms = max(0.0001, float(speech_rms))
        self._noise_profile: np.ndarray | None = None

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, PreprocessStepResult]:
        if audio.size < 512:
            return (audio, self._result(active=True, reason='short-window'))
        window = np.hanning(audio.size).astype(np.float32)
        spectrum = np.fft.rfft(audio * window)
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.exp(1j * np.angle(spectrum))
        rms = _rms(audio)
        if self._noise_profile is None:
            self._noise_profile = magnitude
        elif rms <= self._speech_rms:
            self._noise_profile = (1.0 - self._noise_update) * self._noise_profile + self._noise_update * magnitude
        noise = self._noise_profile
        attenuated = np.maximum(magnitude - noise * self._reduction, magnitude * 0.18)
        out = np.fft.irfft(attenuated * phase, n=audio.size).astype(np.float32)
        out = np.clip(out, -1.0, 1.0)
        return (out, self._result(active=True, reason='processed', metrics={'rms_in': rms, 'rms_out': _rms(out)}))

class AdaptiveGainStage(_BaseStage):
    name = 'adaptive-gain'

    def __init__(self, *, target_rms: float=0.055, max_gain: float=3.5, smoothing: float=0.12) -> None:
        self._target_rms = max(0.001, float(target_rms))
        self._max_gain = max(1.0, float(max_gain))
        self._smoothing = max(0.001, min(1.0, float(smoothing)))
        self._gain = 1.0

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, PreprocessStepResult]:
        rms = _rms(audio)
        if audio.size == 0 or rms <= 1e-05:
            return (audio, self._result(active=True, reason='empty-or-silent', metrics={'gain': self._gain}))
        target_gain = min(self._max_gain, max(0.35, self._target_rms / rms))
        self._gain = (1.0 - self._smoothing) * self._gain + self._smoothing * target_gain
        out = np.clip(audio * self._gain, -1.0, 1.0).astype(np.float32)
        return (out, self._result(active=True, reason='processed', metrics={'rms_in': rms, 'rms_out': _rms(out), 'gain': float(self._gain)}))

class AudioPreprocessingPipeline:
    """Pre-STT processing pipeline that receives raw AudioChunk and returns normalized mono chunk."""

    def __init__(self, stages: list[AudioPreprocessStage], *, target_sample_rate: int=16000) -> None:
        self._stages = list(stages)
        self._target_sample_rate = max(8000, int(target_sample_rate))
        self._last_result: dict[str, object] = {'enabled': bool(self._stages), 'steps': []}

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self._stages]

    @property
    def active_stage_names(self) -> list[str]:
        return [stage.name for stage in self._stages if stage.active]

    def process(self, chunk: AudioChunk, *, channel_mode: str='mono') -> AudioChunk:
        """Apply configured preprocessing stages before VAD/STT consumption."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        steps: list[dict[str, object]] = []
        if audio.size == 0:
            self._last_result = {'enabled': bool(self._stages), 'steps': steps, 'reason': 'empty-audio'}
            return AudioChunk(pcm16=b'', sample_rate=self._target_sample_rate, channels=1)
        for stage in self._stages:
            (audio, result) = stage.process(audio, self._target_sample_rate)
            steps.append({'name': result.name, 'active': result.active, 'reason': result.reason, 'metrics': dict(result.metrics)})
        self._last_result = {'enabled': bool(self._stages), 'steps': steps}
        return AudioChunk(pcm16=to_int16_pcm(audio), sample_rate=self._target_sample_rate, channels=1)

    def get_last_result(self) -> dict[str, object]:
        return dict(self._last_result)

def create_audio_preprocessing_pipeline(config: RuntimeConfig) -> AudioPreprocessingPipeline:
    """Build preprocessing stages from config.preprocess_enabled and config.preprocess_modules."""
    if not bool(config.preprocess_enabled):
        return AudioPreprocessingPipeline([])
    modules = _parse_modules(config.preprocess_modules)
    stages: list[AudioPreprocessStage] = []
    if 'auto' in modules:
        candidates: list[AudioPreprocessStage] = [WebRTCAudioProcessingStage(name='webrtc-ns', enable_ns=True), RNNoiseStage()]
        stages.extend([stage for stage in candidates if stage.active])
        if not stages:
            stages.append(SpectralGateStage())
        stages.append(AdaptiveGainStage())
        return AudioPreprocessingPipeline(stages)
    for module in modules:
        if module == 'none':
            continue
        if module == 'webrtc-ns':
            stages.append(WebRTCAudioProcessingStage(name='webrtc-ns', enable_ns=True))
        elif module == 'webrtc-agc':
            stages.append(WebRTCAudioProcessingStage(name='webrtc-agc', enable_agc=True))
        elif module == 'webrtc-aec':
            stages.append(WebRTCAudioProcessingStage(name='webrtc-aec', enable_aec=True))
        elif module == 'rnnoise':
            stages.append(RNNoiseStage())
        elif module == 'spectral-gate':
            stages.append(SpectralGateStage())
        elif module == 'adaptive-gain':
            stages.append(AdaptiveGainStage())
    return AudioPreprocessingPipeline(stages)

def _parse_modules(raw: str) -> list[str]:
    modules = [item.strip().lower() for item in (raw or 'auto').split(',') if item.strip()]
    return modules or ['auto']

def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
