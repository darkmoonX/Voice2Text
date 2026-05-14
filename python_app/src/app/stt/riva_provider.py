"""NVIDIA Riva gRPC STT provider adapter."""
from __future__ import annotations
import inspect
import time
from typing import Any, Optional
from ..audio_capture import AudioChunk
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample, to_int16_pcm

class RivaGrpcTranscriber:
    """Riva gRPC provider wrapper; sends language_code in RecognitionConfig per chunk."""

    def __init__(self, uri: str='localhost:50051', use_ssl: bool=False, ssl_cert: Optional[str]=None, api_key: Optional[str]=None, default_language_code: str='en-US', target_sample_rate: int=16000) -> None:
        try:
            import riva.client as riva_client
        except Exception as exc:
            raise RuntimeError('nvidia-riva-client is not installed. Run: pip install nvidia-riva-client') from exc
        self._riva_client = riva_client
        self._target_sample_rate = int(target_sample_rate)
        self._default_language_code = default_language_code.strip() or 'en-US'
        self._next_retry_monotonic = 0.0
        self._retry_backoff_seconds = 10.0
        self._reported_unavailable = False
        auth_kwargs: dict[str, object] = {'uri': uri, 'use_ssl': bool(use_ssl)}
        if ssl_cert:
            auth_kwargs['ssl_cert'] = ssl_cert
        metadata_args: list[tuple[str, str]] = []
        if api_key:
            metadata_args.append(('authorization', f'Bearer {api_key}'))
        if metadata_args:
            auth_kwargs['metadata_args'] = metadata_args
        auth = _invoke_with_supported_kwargs(riva_client.Auth, auth_kwargs)
        self._asr = riva_client.ASRService(auth)

    def has_enough_signal(self, chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk: AudioChunk, language: Optional[str]=None, channel_mode: str='mono') -> str:
        """Decode a chunk through Riva offline_recognize with mapped language code."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return ''
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 5:
            return ''
        (normalized_lang, zh_script) = normalize_language_hint(language)
        language_code = self._resolve_riva_language_code(normalized_lang)
        recognition_config = self._build_recognition_config(language_code)
        audio_bytes = to_int16_pcm(audio)
        if self._next_retry_monotonic > time.monotonic():
            return ''
        try:
            try:
                response = self._asr.offline_recognize(audio_bytes, recognition_config)
            except TypeError:
                response = self._asr.offline_recognize(audio_bytes=audio_bytes, config=recognition_config)
        except Exception as exc:
            if self._is_connection_unavailable(exc):
                self._next_retry_monotonic = time.monotonic() + self._retry_backoff_seconds
                if not self._reported_unavailable:
                    self._reported_unavailable = True
                    raise RuntimeError('Riva endpoint is unreachable. Check --riva-uri and Riva server status.') from exc
                return ''
            raise
        text = self._extract_transcript(response)
        return normalize_chinese_script(text, zh_script)

    def _build_recognition_config(self, language_code: str) -> object:
        encoding = self._resolve_audio_encoding()
        kwargs = {'encoding': encoding, 'sample_rate_hertz': self._target_sample_rate, 'language_code': language_code, 'max_alternatives': 1, 'enable_automatic_punctuation': True, 'verbatim_transcripts': False}
        return _invoke_with_supported_kwargs(self._riva_client.RecognitionConfig, kwargs)

    def _resolve_audio_encoding(self) -> object:
        encoding_enum = getattr(self._riva_client, 'AudioEncoding', None)
        if encoding_enum is not None and hasattr(encoding_enum, 'LINEAR_PCM'):
            return encoding_enum.LINEAR_PCM
        return 1

    def _resolve_riva_language_code(self, normalized_lang: Optional[str]) -> str:
        if not normalized_lang:
            return self._default_language_code
        mapping = {'en': 'en-US', 'ja': 'ja-JP', 'ko': 'ko-KR', 'zh': 'zh-CN'}
        return mapping.get(normalized_lang, normalized_lang)

    @staticmethod
    def _is_connection_unavailable(exc: Exception) -> bool:
        message = str(exc).lower()
        return 'statuscode.unavailable' in message or 'failed to connect to all addresses' in message or 'connection refused' in message or ('10061' in message)

    @staticmethod
    def _extract_transcript(response: object) -> str:
        if response is None:
            return ''
        texts: list[str] = []
        results = getattr(response, 'results', None)
        if results is None and isinstance(response, dict):
            results = response.get('results')
        for result in results or []:
            alternatives = getattr(result, 'alternatives', None)
            if alternatives is None and isinstance(result, dict):
                alternatives = result.get('alternatives', [])
            if not alternatives:
                continue
            best = alternatives[0]
            transcript = getattr(best, 'transcript', None)
            if transcript is None and isinstance(best, dict):
                transcript = best.get('transcript', '')
            value = str(transcript or '').strip()
            if value:
                texts.append(value)
        return ' '.join(texts).strip()

def _invoke_with_supported_kwargs(factory: Any, kwargs: dict[str, object]) -> Any:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    if any((param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())):
        return factory(**kwargs)
    filtered = {key: value for (key, value) in kwargs.items() if key in sig.parameters}
    return factory(**filtered)
