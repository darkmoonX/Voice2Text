"""STT provider registry and alias normalization seam."""
from __future__ import annotations

from dataclasses import dataclass

from .base import SUPPORTED_STT_PROVIDERS, STTProvider


@dataclass(frozen=True)
class STTProviderSpec:
    provider: STTProvider
    aliases: tuple[str, ...]
    supports_gpu_variant: bool
    supports_source_language_hint: bool


STT_PROVIDER_SPECS: tuple[STTProviderSpec, ...] = (
    STTProviderSpec(provider='whisper', aliases=('whisper', 'faster-whisper', 'faster_whisper'), supports_gpu_variant=True, supports_source_language_hint=True),
    STTProviderSpec(provider='whisperx', aliases=('whisperx', 'whisper-x', 'whisper_x'), supports_gpu_variant=True, supports_source_language_hint=True),
)


_PROVIDER_ALIAS_MAP: dict[str, STTProvider] = {}
_PROVIDER_SPEC_MAP: dict[STTProvider, STTProviderSpec] = {}
for _spec in STT_PROVIDER_SPECS:
    _PROVIDER_SPEC_MAP[_spec.provider] = _spec
    for _alias in _spec.aliases:
        _PROVIDER_ALIAS_MAP[_alias] = _spec.provider


def normalize_stt_provider(provider: str) -> STTProvider:
    normalized = (provider or '').strip().lower()
    if normalized in _PROVIDER_ALIAS_MAP:
        return _PROVIDER_ALIAS_MAP[normalized]
    supported = ', '.join(SUPPORTED_STT_PROVIDERS)
    raise ValueError(f'Unsupported STT provider: {provider}. Supported providers: {supported}')


def normalize_stt_variant(variant: str) -> str:
    normalized = (variant or 'auto').strip().lower()
    if normalized not in {'auto', 'cpu', 'gpu'}:
        return 'auto'
    return normalized


def provider_supports_gpu_variant(provider: str) -> bool:
    normalized = normalize_stt_provider(provider)
    return _PROVIDER_SPEC_MAP[normalized].supports_gpu_variant


def provider_supports_source_language_hint(provider: str) -> bool:
    normalized = normalize_stt_provider(provider)
    return _PROVIDER_SPEC_MAP[normalized].supports_source_language_hint
