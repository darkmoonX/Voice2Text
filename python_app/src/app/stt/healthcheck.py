"""Provider dependency and model-availability validation routines for pre-run diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import socket
from typing import Callable, Iterable
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12
from ..model_paths import library_model_dir
from .base import SUPPORTED_STT_PROVIDERS, STTProvider
from .registry import normalize_stt_provider, normalize_stt_variant
from .model_assets import find_model_preset, has_model_preset

@dataclass
class ProviderHealthReport:
    provider: STTProvider
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

def run_provider_health_check(config: RuntimeConfig, *, scope: str='all') -> list[ProviderHealthReport]:
    normalized_scope = (scope or 'all').strip().lower()
    providers: Iterable[STTProvider]
    if normalized_scope == 'active':
        providers = [normalize_stt_provider(config.stt_provider)]
    else:
        providers = SUPPORTED_STT_PROVIDERS
    reports: list[ProviderHealthReport] = []
    for provider in providers:
        reports.append(_check_provider(config, provider))
    return reports

def summarize_health_reports(reports: list[ProviderHealthReport]) -> str:
    lines: list[str] = []
    for report in reports:
        status = 'OK' if report.ok else 'FAIL'
        lines.append(f'[{status}] provider={report.provider}')
        for (key, value) in report.details.items():
            lines.append(f'  - {key}: {value}')
        for warning in report.warnings:
            lines.append(f'  - warning: {warning}')
        for issue in report.issues:
            lines.append(f'  - issue: {issue}')
    return '\n'.join(lines)

def has_failed_reports(reports: list[ProviderHealthReport]) -> bool:
    return any((not report.ok for report in reports))

def _check_provider(config: RuntimeConfig, provider: STTProvider) -> ProviderHealthReport:
    checker = _PROVIDER_CHECKERS.get(provider)
    if checker is not None:
        return checker(config)
    report = ProviderHealthReport(provider=provider, ok=False)
    report.issues.append('Unsupported provider')
    return report

def _check_whisper(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='whisper', ok=True)
    try:
        import faster_whisper
    except Exception as exc:
        report.issues.append(f'Python package missing: faster-whisper ({exc})')
    model_ref = (config.stt_model_path or '').strip() or (config.model_size or 'small').strip()
    report.details['model_ref'] = model_ref
    path_like = _is_path_like(model_ref)
    if path_like:
        model_path = Path(model_ref)
        if not model_path.exists():
            report.issues.append(f'Model path does not exist: {model_path}')
        else:
            has_model_bin = (model_path / 'model.bin').exists()
            has_config = (model_path / 'config.json').exists()
            if not has_model_bin or not has_config:
                report.warnings.append('Model folder exists but model.bin/config.json are missing; runtime may fail.')
    else:
        model_root = library_model_dir('faster-whisper')
        local_dir = model_root / model_ref
        if local_dir.exists():
            report.details['resolved_local_model'] = str(local_dir)
        elif not config.stt_auto_download:
            report.issues.append('Model is not in local cache and stt_auto_download is disabled.')
        else:
            report.warnings.append('Model not found in local cache; it will be downloaded when provider starts.')
    if normalize_stt_variant(config.stt_variant) == 'gpu' and (not can_load_cublas12()):
        report.warnings.append('GPU mode requested but cublas64_12.dll is unavailable. Runtime will fallback to CPU unless CUDA compatibility alias can be prepared.')
    report.ok = not report.issues
    return report

def _check_vosk(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='vosk', ok=True)
    try:
        import vosk
    except Exception as exc:
        report.issues.append(f'Python package missing: vosk ({exc})')
    if normalize_stt_variant(config.stt_variant) == 'gpu':
        report.warnings.append('Vosk Python runtime is CPU-only in this integration.')
    model_ref = (config.stt_model_path or '').strip() or (config.model_size or 'small').strip()
    report.details['model_ref'] = model_ref
    if _is_path_like(model_ref):
        model_path = Path(model_ref)
        if not model_path.exists():
            report.issues.append(f'Model path does not exist: {model_path}')
    else:
        model_root = library_model_dir('vosk')
        local_dir = model_root / model_ref
        if local_dir.exists():
            report.details['resolved_local_model'] = str(local_dir)
        elif config.stt_auto_download and has_model_preset('vosk', model_ref):
            report.warnings.append('Preset model missing locally; runtime will try auto-download.')
        elif not config.stt_auto_download:
            report.issues.append('Preset model missing locally and stt_auto_download is disabled.')
        else:
            report.issues.append('Unable to locate model folder. Set stt_model_path or use a supported preset alias.')
    report.ok = not report.issues
    return report

def _check_sherpa_onnx(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='sherpa-onnx', ok=True)
    try:
        import sherpa_onnx
    except Exception as exc:
        report.issues.append(f'Python package missing: sherpa-onnx ({exc})')
    requested_provider = _resolve_sherpa_provider(config)
    report.details['execution_provider'] = requested_provider
    if requested_provider == 'cuda' and (not _has_onnxruntime_cuda_provider()):
        report.warnings.append('CUDAExecutionProvider is unavailable; runtime will fallback to CPU. Install onnxruntime-gpu and rebuild sherpa-onnx with -DSHERPA_ONNX_ENABLE_GPU=ON for GPU support.')
        report.details['effective_execution_provider'] = 'cpu'
    model_ref = (config.stt_model_path or '').strip() or (config.model_size or 'small').strip()
    report.details['model_ref'] = model_ref
    if _is_path_like(model_ref):
        model_path = Path(model_ref)
        if not model_path.exists():
            report.issues.append(f'Model path does not exist: {model_path}')
        elif not _is_valid_sherpa_model_dir(model_path):
            report.issues.append(_sherpa_model_layout_issue())
    else:
        model_root = library_model_dir('sherpa-onnx')
        local_dir = model_root / model_ref
        preset = find_model_preset('sherpa-onnx', model_ref)
        if preset is not None:
            preset_dir = model_root / preset.target_dir_name
            if preset_dir.exists():
                local_dir = preset_dir
        if local_dir.exists():
            if _is_valid_sherpa_model_dir(local_dir):
                report.details['resolved_local_model'] = str(local_dir)
            else:
                report.issues.append(_sherpa_model_layout_issue())
        elif config.stt_auto_download and has_model_preset('sherpa-onnx', model_ref):
            report.warnings.append('Preset model missing locally; runtime will try auto-download.')
        elif not config.stt_auto_download:
            report.issues.append('Preset model missing locally and stt_auto_download is disabled.')
        else:
            report.issues.append('Unable to locate model folder. Set stt_model_path or use a supported preset alias.')
    report.ok = not report.issues
    return report

def _check_riva(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='riva', ok=True)
    try:
        import riva.client
    except Exception as exc:
        report.issues.append(f'Python package missing: nvidia-riva-client ({exc})')
    if normalize_stt_variant(config.stt_variant) in {'cpu', 'gpu'}:
        report.warnings.append('stt_variant has no local effect for Riva; runtime depends on server setup.')
    uri = (config.riva_uri or 'localhost:50051').strip()
    report.details['uri'] = uri
    host = ''
    port = 0
    if ':' not in uri:
        report.issues.append('Invalid Riva URI format; expected host:port')
    else:
        (host, port_raw) = uri.rsplit(':', 1)
        host = host.strip() or 'localhost'
        try:
            port = int(port_raw)
        except ValueError:
            report.issues.append('Invalid Riva URI port')
    if not report.issues and host and (port > 0):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                report.details['connectivity'] = 'reachable'
        except Exception as exc:
            report.issues.append(f'Cannot connect to Riva endpoint: {exc}')
    report.ok = not report.issues
    return report

def _check_funasr(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='funasr', ok=True)
    try:
        import funasr
    except Exception as exc:
        report.issues.append(f'Python package missing: funasr ({exc})')
    model_ref = (config.stt_model_path or '').strip() or (config.model_size or 'small').strip()
    report.details['model_ref'] = model_ref
    if _is_path_like(model_ref):
        model_path = Path(model_ref)
        if not model_path.exists():
            report.issues.append(f'Model path does not exist: {model_path}')
    if normalize_stt_variant(config.stt_variant) == 'gpu':
        try:
            import torch
            if not torch.cuda.is_available():
                report.warnings.append('GPU mode requested but torch.cuda.is_available() is false. Runtime will fallback to CPU.')
        except Exception as exc:
            report.warnings.append(f'Unable to verify CUDA capability via torch: {exc}')
    report.ok = not report.issues
    return report

def _resolve_sherpa_provider(config: RuntimeConfig) -> str:
    variant = normalize_stt_variant(config.stt_variant)
    if variant == 'cpu':
        return 'cpu'
    if variant == 'gpu':
        return 'cuda'
    return (config.sherpa_onnx_provider or 'cpu').strip() or 'cpu'

@lru_cache(maxsize=1)
def _has_onnxruntime_cuda_provider() -> bool:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        return False
    return 'CUDAExecutionProvider' in providers

def _is_path_like(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if path.is_absolute():
        return True
    return any((sep in value for sep in ('/', '\\'))) or value.startswith('.')

def _is_valid_sherpa_model_dir(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_tokens = (model_dir / 'tokens.txt').exists()
    if not has_tokens:
        return False
    has_transducer = all(((model_dir / name).exists() for name in ('encoder.onnx', 'decoder.onnx', 'joiner.onnx')))
    if has_transducer:
        return True
    has_paraformer = (model_dir / 'model.onnx').exists() or (model_dir / 'model.int8.onnx').exists()
    return has_paraformer

def _sherpa_model_layout_issue() -> str:
    return 'Model folder missing required files. Expected either encoder/decoder/joiner/tokens (transducer) or model(.int8).onnx + tokens (paraformer).'


_PROVIDER_CHECKERS: dict[STTProvider, Callable[[RuntimeConfig], ProviderHealthReport]] = {
    'whisper': _check_whisper,
    'vosk': _check_vosk,
    'sherpa-onnx': _check_sherpa_onnx,
    'riva': _check_riva,
    'funasr': _check_funasr,
}
