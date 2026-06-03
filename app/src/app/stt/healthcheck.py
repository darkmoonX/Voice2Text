"""WhisperX dependency and model-availability validation routines for pre-run diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12
from .base import SUPPORTED_STT_PROVIDERS, STTProvider
from .registry import normalize_stt_provider, normalize_stt_variant

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

def _check_whisperx(config: RuntimeConfig) -> ProviderHealthReport:
    report = ProviderHealthReport(provider='whisperx', ok=True)
    try:
        import whisperx
    except Exception as exc:
        report.issues.append(f'Python package missing: whisperx ({exc})')
    model_ref = (config.stt_model_path or '').strip() or (config.model_size or 'small').strip()
    report.details['model_ref'] = model_ref
    report.details['forced_alignment'] = 'on' if bool(getattr(config, 'whisperx_enable_forced_alignment', True)) else 'off'
    report.details['vad'] = 'on' if bool(getattr(config, 'whisperx_enable_vad', True)) else 'off'
    report.details['diarization'] = 'on' if bool(getattr(config, 'whisperx_enable_diarization', False)) else 'off'
    if bool(getattr(config, 'whisperx_enable_diarization', False)) and (not str(getattr(config, 'whisperx_hf_token', '') or '').strip()):
        report.warnings.append('WhisperX diarization is enabled but HF token is empty. pyannote model download/access may fail.')
    if _is_path_like(model_ref):
        model_path = Path(model_ref)
        if not model_path.exists():
            report.issues.append(f'Model path does not exist: {model_path}')
    if normalize_stt_variant(config.stt_variant) == 'gpu':
        if not can_load_cublas12():
            report.warnings.append('GPU mode requested but cublas64_12.dll is unavailable. Runtime may fallback to CPU.')
    report.ok = not report.issues
    return report

def _is_path_like(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if path.is_absolute():
        return True
    return any((sep in value for sep in ('/', '\\'))) or value.startswith('.')

_PROVIDER_CHECKERS: dict[STTProvider, Callable[[RuntimeConfig], ProviderHealthReport]] = {
    'whisperx': _check_whisperx,
}
