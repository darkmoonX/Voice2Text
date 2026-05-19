"""Provider-related settings schema rules."""
from __future__ import annotations
from ..stt.registry import normalize_stt_provider, provider_supports_gpu_variant, provider_supports_source_language_hint


def allowed_stt_variants(provider: str) -> list[str]:
    if provider_supports_gpu_variant(provider):
        return ['auto', 'cpu', 'gpu']
    return ['auto']


def default_stt_model(provider: str) -> str:
    normalized = normalize_stt_provider(provider)
    mapping = {
        'whisper': 'small',
        'whisperx': 'small',
    }
    return mapping.get(normalized, 'small')


def is_path_like(value: str) -> bool:
    if not value:
        return False
    return any((ch in value for ch in ('/', '\\'))) or value.startswith('.')


def provider_supports_source_language(provider: str) -> bool:
    return provider_supports_source_language_hint(provider)
