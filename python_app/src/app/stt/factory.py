"""STT provider factory: resolves runtime variant/device and constructs provider transcribers."""
from __future__ import annotations
from functools import lru_cache
from typing import Callable, Optional
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12, ensure_cublas12_from_source
from .base import STTProvider, STTTranscriber
from .funasr_provider import FunASRTranscriber
from .riva_provider import RivaGrpcTranscriber
from .sherpa_onnx_provider import SherpaOnnxTranscriber
from .vosk_provider import VoskTranscriber
from .whisper_provider import FasterWhisperTranscriber
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

def _resolve_sherpa_provider(config: RuntimeConfig, progress_callback: Callable[[str], None] | None) -> str:
    variant = normalize_stt_variant(config.stt_variant)
    requested = (config.sherpa_onnx_provider or 'cpu').strip() or 'cpu'
    if variant == 'cpu':
        requested = 'cpu'
    elif variant == 'gpu':
        requested = 'cuda'
    if requested.lower() == 'cuda' and (not _has_onnxruntime_cuda_provider()):
        _emit_progress(progress_callback, 'Sherpa-ONNX GPU requested but CUDAExecutionProvider is unavailable. Falling back to CPU. If GPU is required, install onnxruntime-gpu and rebuild sherpa-onnx with -DSHERPA_ONNX_ENABLE_GPU=ON.')
        return 'cpu'
    return requested

def _resolve_funasr_device(config: RuntimeConfig, progress_callback: Callable[[str], None] | None) -> str:
    variant = normalize_stt_variant(config.stt_variant)
    requested: str
    if variant == 'cpu':
        requested = 'cpu'
    elif variant == 'gpu':
        requested = 'cuda:0'
    else:
        configured = (config.funasr_device or '').strip()
        if configured:
            requested = configured
        elif (config.model_device or '').lower().startswith('cuda'):
            requested = 'cuda:0'
        else:
            requested = 'cpu'
    if requested.lower().startswith('cuda') and (not _has_torch_cuda()):
        _emit_progress(progress_callback, 'FunASR GPU requested but Torch CUDA is unavailable in this environment. Falling back to CPU.')
        return 'cpu'
    return requested

def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)

@lru_cache(maxsize=1)
def _has_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

@lru_cache(maxsize=1)
def _has_onnxruntime_cuda_provider() -> bool:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        return False
    return 'CUDAExecutionProvider' in providers

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


def _build_vosk(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    del device_override, compute_type_override
    auto_download = bool(config.stt_auto_download)
    model_ref = config.stt_model_path.strip() or config.model_size
    return VoskTranscriber(model_ref=model_ref, auto_download=auto_download, progress_callback=progress_callback)


def _build_sherpa_onnx(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    del device_override, compute_type_override
    auto_download = bool(config.stt_auto_download)
    model_ref = config.stt_model_path.strip() or config.model_size
    return SherpaOnnxTranscriber(model_ref=model_ref, provider=_resolve_sherpa_provider(config, progress_callback), auto_download=auto_download, progress_callback=progress_callback)


def _build_riva(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    del device_override, compute_type_override, progress_callback
    return RivaGrpcTranscriber(uri=config.riva_uri, use_ssl=config.riva_use_ssl, ssl_cert=config.riva_ssl_cert or None, api_key=config.riva_api_key or None, default_language_code=config.riva_language_code)


def _build_funasr(config: RuntimeConfig, *, device_override: Optional[str], compute_type_override: Optional[str], progress_callback: Callable[[str], None] | None) -> STTTranscriber:
    del device_override, compute_type_override
    auto_download = bool(config.stt_auto_download)
    model_ref = config.stt_model_path.strip() or config.model_size
    return FunASRTranscriber(model_ref=model_ref, device=_resolve_funasr_device(config, progress_callback), vad_model=config.funasr_vad_model, auto_download=auto_download, progress_callback=progress_callback)


_PROVIDER_BUILDERS: dict[STTProvider, Callable[..., STTTranscriber]] = {
    'whisper': _build_whisper,
    'vosk': _build_vosk,
    'sherpa-onnx': _build_sherpa_onnx,
    'riva': _build_riva,
    'funasr': _build_funasr,
}
