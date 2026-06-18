"""WhisperX dependency and model-availability validation routines for pre-run diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Callable, Iterable
from ..config import RuntimeConfig
from ..cuda_compat import can_load_cublas12
from .base import SUPPORTED_STT_PROVIDERS, STTProvider
from .model_cache import cache_summary, human_size
from .registry import normalize_stt_provider, normalize_stt_variant


@dataclass
class HealthCheck:
    """One structured, actionable check row (for the Phase B wizard + the CLI)."""
    id: str
    label: str
    status: str          # "ok" | "warn" | "fail"
    detail: str = ""
    fix_hint: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "fix_hint": self.fix_hint,
        }


@dataclass
class ProviderHealthReport:
    provider: STTProvider
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)
    checks: list[HealthCheck] = field(default_factory=list)

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
        for check in report.checks:
            line = f'  - check[{check.status}] {check.id}: {check.detail or check.label}'
            if check.status != 'ok' and check.fix_hint:
                line += f' -> fix: {check.fix_hint}'
            lines.append(line)
    return '\n'.join(lines)

def has_failed_reports(reports: list[ProviderHealthReport]) -> bool:
    return any((not report.ok for report in reports))


# --- Structured, individually-testable checks (CUDA / FFmpeg / HF token / bridge / cache) ---

def check_cuda(config: RuntimeConfig, *, cublas_probe: Callable[[], bool] = can_load_cublas12) -> HealthCheck:
    if normalize_stt_variant(config.stt_variant) != 'gpu':
        return HealthCheck('cuda', 'CUDA / cuBLAS', 'ok', detail='CPU variant selected; CUDA not required.')
    if cublas_probe():
        return HealthCheck('cuda', 'CUDA / cuBLAS', 'ok', detail='cublas64_12.dll loadable.')
    return HealthCheck(
        'cuda', 'CUDA / cuBLAS', 'warn',
        detail='GPU variant requested but cublas64_12.dll is unavailable; runtime may fall back to CPU.',
        fix_hint='Install the CUDA 12 runtime, or switch the STT variant to CPU.',
    )


def _resolve_ffmpeg(config: RuntimeConfig, which: Callable[[str], str | None]) -> str:
    ffmpeg_dir = str(getattr(config, 'ffmpeg_dll_dir', '') or '').strip()
    if ffmpeg_dir:
        for name in ('ffmpeg.exe', 'ffmpeg'):
            candidate = Path(ffmpeg_dir) / name
            if candidate.exists():
                return str(candidate)
    found = which('ffmpeg')
    return str(found or '')


def check_ffmpeg(config: RuntimeConfig, *, which: Callable[[str], str | None] = shutil.which) -> HealthCheck:
    resolved = _resolve_ffmpeg(config, which)
    if resolved:
        return HealthCheck('ffmpeg', 'FFmpeg', 'ok', detail=f'Found: {resolved}')
    return HealthCheck(
        'ffmpeg', 'FFmpeg', 'warn',
        detail='ffmpeg not found on PATH or in the configured FFmpeg dir; file import/decode will fail.',
        fix_hint='Install FFmpeg and add it to PATH, or set the FFmpeg directory in Settings (--ffmpeg-dll-dir).',
    )


def check_hf_token(config: RuntimeConfig) -> HealthCheck:
    token = str(getattr(config, 'whisperx_hf_token', '') or '').strip()
    diarization_on = bool(getattr(config, 'whisperx_enable_diarization', False))
    if not diarization_on:
        return HealthCheck('hf_token', 'HuggingFace token', 'ok', detail='Diarization off; HF token not required.')
    if token:
        # Never echo the token; report presence only.
        return HealthCheck('hf_token', 'HuggingFace token', 'ok', detail='Token present (redacted).')
    return HealthCheck(
        'hf_token', 'HuggingFace token', 'warn',
        detail='Diarization is enabled but no HF token is set; pyannote model download/access may fail.',
        fix_hint='Paste a HuggingFace access token in Settings (accept the pyannote model terms first).',
    )


def check_capture_bridge(
    *,
    resolve: Callable[[], Path | None] | None = None,
    health: Callable[[Path], tuple[bool, str]] | None = None,
) -> HealthCheck:
    # Imported lazily so the health check has no hard dependency on the capture package at import time.
    if resolve is None or health is None:
        from ..capture.bridge_probe import check_bridge_health, resolve_capture_bridge_executable
        resolve = resolve or resolve_capture_bridge_executable
        health = health or check_bridge_health
    exe = resolve()
    if exe is None:
        return HealthCheck(
            'capture_bridge', 'C++ capture bridge', 'warn',
            detail='Bridge executable not found; capture will use the pure-Python WASAPI fallback.',
            fix_hint='Build it via app/native/audio_bridge/build_bridge.ps1, or set VOICE2TEXT_CPP_CAPTURE_BRIDGE.',
        )
    healthy, reason = health(Path(exe))
    if healthy:
        return HealthCheck('capture_bridge', 'C++ capture bridge', 'ok', detail=f'{exe} ({reason})')
    return HealthCheck(
        'capture_bridge', 'C++ capture bridge', 'warn',
        detail=f'Bridge present but unhealthy ({reason}); capture will use the Python WASAPI fallback.',
        fix_hint='Rebuild the bridge (build_bridge.ps1); check for missing runtime DLL dependencies.',
    )


def check_model_cache(config: RuntimeConfig, *, summary: Callable[..., dict[str, object]] = cache_summary) -> HealthCheck:
    try:
        info = summary()
    except Exception as exc:
        return HealthCheck('model_cache', 'Model / alignment cache', 'warn', detail=f'Cache scan failed: {exc}')
    total = int(info.get('total_bytes', 0) or 0)
    count = int(info.get('entry_count', 0) or 0)
    if count <= 0:
        return HealthCheck(
            'model_cache', 'Model / alignment cache', 'warn',
            detail='No cached WhisperX models found; first run will download models.',
            fix_hint='Predownload the base + alignment models, or just allow the first run to fetch them.',
        )
    return HealthCheck(
        'model_cache', 'Model / alignment cache', 'ok',
        detail=f'{count} cached model folder(s), {human_size(total)} total.',
    )


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
    # Structured, actionable checks (new in round 0022). These are informational (ok/warn only) and do
    # not flip report.ok, so the CLI exit code stays back-compat with the issue-driven contract.
    report.checks = [
        check_cuda(config),
        check_ffmpeg(config),
        check_hf_token(config),
        check_capture_bridge(),
        check_model_cache(config),
    ]
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
