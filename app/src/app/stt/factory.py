"""WhisperX provider factory: resolves runtime variant/device and constructs transcribers."""
from __future__ import annotations
from typing import Callable, Optional
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12, ensure_cublas12_from_source
from .base import STTProvider, STTTranscriber
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


def _whisperx_needs_torch_cuda(config: RuntimeConfig) -> bool:
    return any(
        (
            bool(getattr(config, 'whisperx_enable_forced_alignment', True)),
            bool(getattr(config, 'whisperx_enable_vad', True)),
            bool(getattr(config, 'whisperx_enable_diarization', False)),
            bool(getattr(config, 'whisperx_speaker_profile_enabled', True)),
        )
    )

def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)

def create_stt_transcriber(config: RuntimeConfig, *, device_override: Optional[str]=None, compute_type_override: Optional[str]=None, progress_callback: Callable[[str], None] | None=None) -> STTTranscriber:
    """Build the WhisperX STT engine instance used by the controller."""
    provider = normalize_stt_provider(config.stt_provider)
    if provider != 'whisperx':
        raise ValueError(f'Unsupported STT provider: {provider}')
    return _build_whisperx(config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=progress_callback)


def _build_whisperx(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    from .whisperx_provider import WhisperXTranscriber

    model_ref = config.stt_model_path.strip() or config.model_size
    (device, compute_type) = _resolve_whisper_runtime(config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=progress_callback)
    if str(device).lower().startswith('cuda') and (not _has_torch_cuda()):
        if _whisperx_needs_torch_cuda(config):
            _emit_progress(progress_callback, 'WhisperX CUDA unavailable in current torch build. Falling back to CPU int8.')
            device = 'cpu'
            compute_type = 'int8'
        else:
            _emit_progress(progress_callback, 'Torch CUDA unavailable, but WhisperX is running ASR-only mode. Keep device=cuda for CTranslate2 ASR.')
    return WhisperXTranscriber(
        model_ref=model_ref,
        device=device,
        compute_type=compute_type,
        beam_size=max(1, int(config.whisper_beam_size or 5)),
        batch_size=max(1, int(getattr(config, 'whisper_batch_size', 4) or 4)),
        enable_phoneme_asr=bool(getattr(config, 'whisperx_enable_phoneme_asr', True)),
        enable_forced_alignment=bool(getattr(config, 'whisperx_enable_forced_alignment', True)),
        enable_vad=bool(getattr(config, 'whisperx_enable_vad', True)),
        vad_method=str(getattr(config, 'whisperx_vad_method', 'silero-vad') or 'silero-vad'),
        enable_diarization=bool(getattr(config, 'whisperx_enable_diarization', False)),
        alignment_model=str(getattr(config, 'whisperx_alignment_model', '') or ''),
        alignment_language=str(getattr(config, 'whisperx_alignment_language', 'auto') or 'auto'),
        alignment_device=str(getattr(config, 'whisperx_alignment_device', 'auto') or 'auto'),
        diarization_device=str(getattr(config, 'whisperx_diarization_device', 'auto') or 'auto'),
        source_language_hint=str(getattr(config, 'source_language', '') or ''),
        diarization_model=str(getattr(config, 'whisperx_diarization_model', 'pyannote/speaker-diarization-3.1') or 'pyannote/speaker-diarization-3.1'),
        hf_token=str(getattr(config, 'whisperx_hf_token', '') or ''),
        speaker_profile_enabled=bool(getattr(config, 'whisperx_speaker_profile_enabled', True)),
        speaker_profile_backend=str(getattr(config, 'whisperx_speaker_profile_backend', 'pyannote') or 'pyannote'),
        speaker_profile_model=str(getattr(config, 'whisperx_speaker_profile_model', 'pyannote/embedding') or 'pyannote/embedding'),
        speaker_speechbrain_model=str(getattr(config, 'whisperx_speaker_speechbrain_model', 'speechbrain/spkrec-ecapa-voxceleb') or 'speechbrain/spkrec-ecapa-voxceleb'),
        speaker_nemo_model=str(getattr(config, 'whisperx_speaker_nemo_model', 'nvidia/speakerverification_en_titanet_large') or 'nvidia/speakerverification_en_titanet_large'),
        speaker_profile_match_threshold=float(getattr(config, 'whisperx_speaker_profile_match_threshold', 0.72) or 0.72),
        speaker_profile_min_seconds=float(getattr(config, 'whisperx_speaker_profile_min_seconds', 0.8) or 0.8),
        speaker_profile_reconcile_threshold=float(getattr(config, 'whisperx_speaker_profile_reconcile_threshold', 0.52) or 0.52),
        speaker_profile_store_path=str(getattr(config, 'whisperx_speaker_profile_store_path', '') or ''),
        speaker_marker_style=str(getattr(config, 'speaker_marker_style', 'spk') or 'spk'),
        speaker_pause_break_seconds=float(getattr(config, 'speaker_pause_break_seconds', 1.8)),
        auto_download=bool(config.stt_auto_download),
        progress_callback=progress_callback,
    )


