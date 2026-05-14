"""Voice activity detection stages and pipeline used before STT transcription."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from ..audio_capture import AudioChunk
from .base import STTProvider
from .audio_utils import pcm16_to_mono_float, resample
if TYPE_CHECKING:
    from ..config import RuntimeConfig

@dataclass(frozen=True)
class VADStepResult:
    name: str
    passed: bool
    reason: str
    metrics: dict[str, float]

class VADStage:
    name = 'base'

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> VADStepResult:
        raise NotImplementedError

class RMSGate(VADStage):
    name = 'rms-gate'

    def __init__(self, threshold: float) -> None:
        self._threshold = max(0.0, float(threshold))

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> VADStepResult:
        if audio.size == 0:
            return VADStepResult(name=self.name, passed=False, reason='empty-audio', metrics={'rms': 0.0, 'threshold': self._threshold})
        rms = float(np.sqrt(np.mean(np.square(audio))))
        passed = rms >= self._threshold
        return VADStepResult(name=self.name, passed=passed, reason='rms-pass' if passed else 'rms-below-threshold', metrics={'rms': rms, 'threshold': self._threshold})

class AdaptiveRMSGate(VADStage):
    name = 'adaptive-rms-gate'

    def __init__(self, *, base_threshold: float, min_threshold: float=0.004, max_threshold: float=0.08, noise_multiplier: float=2.6, margin: float=0.002) -> None:
        self._base_threshold = max(0.0, float(base_threshold))
        self._min_threshold = max(0.0, float(min_threshold))
        self._max_threshold = max(self._min_threshold, float(max_threshold))
        self._noise_multiplier = max(1.0, float(noise_multiplier))
        self._margin = max(0.0, float(margin))
        self._noise_floor: float | None = None

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> VADStepResult:
        if audio.size == 0:
            threshold = self._current_threshold()
            return VADStepResult(name=self.name, passed=False, reason='empty-audio', metrics={'rms': 0.0, 'threshold': threshold, 'noise_floor': float(self._noise_floor or 0.0)})
        rms = float(np.sqrt(np.mean(np.square(audio))))
        threshold = self._current_threshold()
        passed = rms >= threshold
        self._update_noise_floor(rms, passed=passed)
        threshold = self._current_threshold()
        passed = rms >= threshold
        return VADStepResult(name=self.name, passed=passed, reason='adaptive-rms-pass' if passed else 'adaptive-rms-below-threshold', metrics={'rms': rms, 'threshold': threshold, 'noise_floor': float(self._noise_floor or 0.0), 'base_threshold': self._base_threshold})

    def _current_threshold(self) -> float:
        floor = float(self._noise_floor or 0.0)
        dynamic = floor * self._noise_multiplier + self._margin
        threshold = max(self._base_threshold, self._min_threshold, dynamic)
        return min(self._max_threshold, threshold)

    def _update_noise_floor(self, rms: float, *, passed: bool) -> None:
        if self._noise_floor is None:
            initial_cap = max(self._min_threshold, self._base_threshold * 0.75)
            self._noise_floor = min(rms, initial_cap)
            return
        floor = max(1e-06, self._noise_floor)
        if not passed or rms <= floor * 1.55:
            alpha = 0.3 if rms < floor else 0.12
        elif rms <= max(self._base_threshold * 2.0, floor * 2.4):
            alpha = 0.04
        else:
            alpha = 0.01
        self._noise_floor = (1.0 - alpha) * floor + alpha * rms

class SherpaNoiseGuard(VADStage):
    name = 'sherpa-noise-guard'

    def __init__(self, *, flatness_threshold: float=0.62, zcr_threshold: float=0.04, min_samples: int=2048) -> None:
        self._flatness_threshold = float(flatness_threshold)
        self._zcr_threshold = float(zcr_threshold)
        self._min_samples = int(min_samples)

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> VADStepResult:
        if audio.size < self._min_samples:
            return VADStepResult(name=self.name, passed=True, reason='short-window', metrics={'samples': float(audio.size)})
        window = audio[-65536:] if audio.size > 65536 else audio
        signed = np.signbit(window).astype(np.int8)
        zcr = float(np.mean(np.abs(np.diff(signed))))
        spectrum = np.abs(np.fft.rfft(window)) + 1e-10
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        looks_like_noise = flatness >= self._flatness_threshold and zcr >= self._zcr_threshold
        return VADStepResult(name=self.name, passed=not looks_like_noise, reason='noise-guard-reject' if looks_like_noise else 'noise-guard-pass', metrics={'flatness': flatness, 'flatness_threshold': self._flatness_threshold, 'zcr': zcr, 'zcr_threshold': self._zcr_threshold})

class VADPipeline:
    """Pre-transcription decision pipeline that decides whether a chunk should be sent to STT."""

    def __init__(self, stages: list[VADStage], *, target_sample_rate: int=16000) -> None:
        self._stages = list(stages)
        self._target_sample_rate = max(8000, int(target_sample_rate))
        self._last_result: dict[str, object] = {'passed': True, 'reason': 'init', 'steps': []}

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self._stages]

    def should_process(self, chunk: AudioChunk, *, channel_mode: str='mono') -> bool:
        """Return True when all VAD stages pass for the current audio chunk."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            self._last_result = {'passed': False, 'reason': 'empty-audio', 'steps': []}
            return False
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size == 0:
            self._last_result = {'passed': False, 'reason': 'resample-empty', 'steps': []}
            return False
        steps: list[dict[str, object]] = []
        passed = True
        reason = 'vad-pass'
        for stage in self._stages:
            result = stage.evaluate(audio, self._target_sample_rate)
            steps.append({'name': result.name, 'passed': result.passed, 'reason': result.reason, 'metrics': dict(result.metrics)})
            if not result.passed:
                passed = False
                reason = result.reason
                break
        self._last_result = {'passed': passed, 'reason': reason, 'steps': steps}
        return passed

    def get_last_result(self) -> dict[str, object]:
        return dict(self._last_result)

def create_vad_pipeline(config: RuntimeConfig, provider: STTProvider) -> VADPipeline:
    """Build VAD stage list from runtime config and active provider."""
    if not bool(config.vad_enabled):
        return VADPipeline([])
    if bool(getattr(config, 'vad_adaptive_enabled', True)):
        stages: list[VADStage] = [AdaptiveRMSGate(base_threshold=config.vad_rms_threshold, min_threshold=getattr(config, 'vad_adaptive_min_threshold', 0.004), max_threshold=getattr(config, 'vad_adaptive_max_threshold', 0.08), noise_multiplier=getattr(config, 'vad_adaptive_noise_multiplier', 2.6), margin=getattr(config, 'vad_adaptive_margin', 0.002))]
    else:
        stages = [RMSGate(config.vad_rms_threshold)]
    if provider == 'sherpa-onnx' and bool(config.vad_sherpa_noise_guard):
        stages.append(SherpaNoiseGuard())
    return VADPipeline(stages)
