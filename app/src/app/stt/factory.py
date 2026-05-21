"""STT provider factory: resolves runtime variant/device and constructs provider transcribers."""
from __future__ import annotations
from typing import Callable, Optional
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12, ensure_cublas12_from_source
from .base import STTProvider, STTTranscriber
from .whisper_provider import FasterWhisperTranscriber
from .whisperx_provider import WhisperXTranscriber
from .registry import normalize_stt_provider, normalize_stt_variant

def _resolve_whisper_runtime(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> tuple[str, str]:
    device = device_override or config.model_device
    compute_type = compute_type_override or config.compute_type
    if device_override is None and compute_type_override is None:
        variant = normalize_stt_variant(config.stt_variant)
        if variant == 'cpu':
            (device, compute_type) = ('cpu', 'int8')
        elif variant == 'gpu':
            if compute_type.lower() == 'int8':
                compute_type = 'int8_float16'
            device = 'cuda'
    if str(device).lower().startswith('cuda'):
        if not can_load_cublas12():
            alias_ready = ensure_cublas12_from_source(config.cuda_compat_source_dll, compat_dir=getattr(config, 'cuda_compat_dir', None), on_status=progress_callback)
            if alias_ready:
                _emit_progress(progress_callback, 'Whisper CUDA compatibility alias prepared before startup.')
            else:
                _emit_progress(progress_callback, 'Whisper CUDA runtime is unavailable on this environment. Falling back to CPU int8.')
                return ('cpu', 'int8')
    return (device, compute_type)


def _has_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)

def create_stt_transcriber(config: RuntimeConfig, *, device_override: Optional[str]=None, compute_type_override: Optional[str]=None, progress_callback: Callable[[str], None] | None=None) -> STTTranscriber:
    """Provider router called by controller to build the active STT engine instance."""
    provider = normalize_stt_provider(config.stt_provider)
    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f'Unsupported STT provider: {provider}')
    return builder(config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=progress_callback)


def _build_whisper(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    auto_download = bool(config.stt_auto_download)
    model_ref = config.stt_model_path.strip() or config.model_size
    (device, compute_type) = _resolve_whisper_runtime(config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=progress_callback)
    return FasterWhisperTranscriber(model_size=model_ref, device=device, compute_type=compute_type, auto_download=auto_download, progress_callback=progress_callback, max_context=config.whisper_max_context, entropy_thold=config.whisper_entropy_thold, logprob_thold=config.whisper_logprob_thold, no_speech_thold=config.whisper_no_speech_thold, temperature=config.whisper_temperature, beam_size=config.whisper_beam_size, best_of=config.whisper_best_of)


def _build_whisperx(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    model_ref = config.stt_model_path.strip() or config.model_size
    (device, compute_type) = _resolve_whisper_runtime(config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=progress_callback)
    if str(device).lower().startswith('cuda') and (not _has_torch_cuda()):
        _emit_progress(progress_callback, 'WhisperX CUDA unavailable in current torch build. Falling back to CPU int8.')
        device = 'cpu'
        compute_type = 'int8'
    return WhisperXTranscriber(
        model_ref=model_ref,
        device=device,
        compute_type=compute_type,
        batch_size=max(1, min(16, int(config.whisper_beam_size or 4))),
        enable_phoneme_asr=bool(getattr(config, 'whisperx_enable_phoneme_asr', True)),
        enable_forced_alignment=bool(getattr(config, 'whisperx_enable_forced_alignment', True)),
        enable_vad=bool(getattr(config, 'whisperx_enable_vad', True)),
        vad_method=str(getattr(config, 'whisperx_vad_method', 'silero-vad') or 'silero-vad'),
        enable_diarization=bool(getattr(config, 'whisperx_enable_diarization', False)),
        alignment_model=str(getattr(config, 'whisperx_alignment_model', '') or ''),
        alignment_language=str(getattr(config, 'whisperx_alignment_language', 'auto') or 'auto'),
        source_language_hint=str(getattr(config, 'source_language', '') or ''),
        diarization_model=str(getattr(config, 'whisperx_diarization_model', 'pyannote/speaker-diarization-3.1') or 'pyannote/speaker-diarization-3.1'),
        hf_token=str(getattr(config, 'whisperx_hf_token', '') or ''),
        auto_download=bool(config.stt_auto_download),
        progress_callback=progress_callback,
    )


_PROVIDER_BUILDERS: dict[STTProvider, Callable[..., STTTranscriber]] = {
    'whisper': _build_whisper,
    'whisperx': _build_whisperx,
}


