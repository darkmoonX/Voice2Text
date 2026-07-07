from voice2text.settings.schema import allowed_stt_variants, default_stt_model, provider_supports_source_language
from voice2text.stt.registry import (
    normalize_stt_provider,
    normalize_stt_variant,
    provider_supports_gpu_variant,
    provider_supports_source_language_hint,
)


def test_provider_alias_normalization():
    assert normalize_stt_provider('whisper') == 'whisperx'
    assert normalize_stt_provider('faster-whisper') == 'whisperx'
    assert normalize_stt_provider('whisper-x') == 'whisperx'
    assert normalize_stt_provider('whispercpp') == 'whispercpp'
    assert normalize_stt_provider('whisper.cpp') == 'whispercpp'


def test_variant_normalization():
    assert normalize_stt_variant('GPU') == 'gpu'
    assert normalize_stt_variant('cpu') == 'cpu'
    assert normalize_stt_variant('unknown') == 'auto'


def test_schema_uses_shared_registry_capabilities():
    providers = ['whisper', 'whisperx', 'whispercpp']
    for provider in providers:
        gpu_variants = allowed_stt_variants(provider)
        if provider_supports_gpu_variant(provider):
            assert gpu_variants == ['auto', 'cpu', 'gpu']
        else:
            assert gpu_variants == ['auto']
        assert provider_supports_source_language(provider) == provider_supports_source_language_hint(provider)


def test_default_model_mapping_stable():
    assert default_stt_model('whisperx') == 'small'
    assert default_stt_model('whispercpp') == 'medium'
