"""FunASR provider integration with model id normalization and download progress emission."""
from __future__ import annotations
import inspect
import os
from pathlib import Path
import threading
import time
import tempfile
from typing import Any, Optional
from urllib.parse import unquote, urlparse
import wave
import numpy as np
from ..audio_capture import AudioChunk
from ..model_paths import library_model_dir
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample
from .model_download import emit_progress, format_download_progress

class FunASRTranscriber:
    """FunASR provider wrapper with model id aliases and progress reporting."""
    _MODEL_ALIAS: dict[str, str] = {
        'small': 'iic/SenseVoiceSmall',
        'sensevoice': 'iic/SenseVoiceSmall',
        'sensevoice-small': 'iic/SenseVoiceSmall',
        'sensevoicesmall': 'iic/SenseVoiceSmall',
        'paraformer-zh': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'paraformer-zh-streaming': 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
        'paraformer-en': 'damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020',
        'conformer-en': 'damo/speech_conformer_asr-en-16k-vocab4199-pytorch',
        'large': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'paraformer-large': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'large-zh': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'speech-paraformer-large-asr-nat-zh-cn-16k-common-vocab8404-pytorch': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'iic/speech-paraformer-large-asr-nat-zh-cn-16k-common-vocab8404-pytorch': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'funasr-nano': 'FunAudioLLM/Fun-ASR-Nano-2512',
        'fun-asr-nano-2512': 'FunAudioLLM/Fun-ASR-Nano-2512',
        'nano': 'FunAudioLLM/Fun-ASR-Nano-2512',
    }

    def __init__(self, model_ref: str='small', device: str='cpu', vad_model: str='fsmn-vad', target_sample_rate: int=16000, auto_download: bool=True, progress_callback=None) -> None:
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError('funasr is not installed. Run: pip install funasr modelscope') from exc
        self._target_sample_rate = int(target_sample_rate)
        self._postprocess = None
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            self._postprocess = rich_transcription_postprocess
        except Exception:
            self._postprocess = None
        normalized_model = self._normalize_model_ref(model_ref)
        prepared_model = self._prepare_model_ref(normalized_model, auto_download=auto_download, progress_callback=progress_callback)
        model_kwargs: dict[str, object] = {'model': normalized_model, 'device': device, 'disable_update': True}
        if prepared_model is not None:
            model_kwargs['model'] = prepared_model
        if vad_model.strip():
            model_kwargs['vad_model'] = vad_model.strip()
            model_kwargs['vad_kwargs'] = {'max_single_segment_time': 30000}
        self._model = self._build_model_with_progress(
            AutoModel,
            model_kwargs,
            auto_download=auto_download,
            progress_callback=progress_callback,
            model_name=str(model_kwargs.get('model', normalized_model)),
        )

    def has_enough_signal(self, chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk: AudioChunk, language: Optional[str]=None, channel_mode: str='mono') -> str:
        """Decode a chunk with FunASR AutoModel and pass language hint when supported."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return ''
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 5:
            return ''
        (normalized_lang, zh_script) = normalize_language_hint(language)
        funasr_lang = self._to_funasr_language(normalized_lang)
        result = self._run_generate(audio, funasr_lang)
        text = self._extract_text(result)
        if text and self._postprocess is not None:
            try:
                text = str(self._postprocess(text) or '').strip()
            except Exception:
                pass
        return normalize_chinese_script(text, zh_script)

    def _run_generate(self, audio: np.ndarray, language: Optional[str]) -> object:
        kwargs: dict[str, object] = {'input': audio, 'cache': {}, 'batch_size_s': 0, 'use_itn': True}
        if language:
            kwargs['language'] = language
        try:
            return _invoke_with_supported_kwargs(self._model.generate, kwargs)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            try:
                self._write_wav(temp_path, audio)
                kwargs['input'] = str(temp_path)
                return _invoke_with_supported_kwargs(self._model.generate, kwargs)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _normalize_model_ref(self, model_ref: str) -> str:
        cleaned = model_ref.strip()
        if not cleaned:
            cleaned = 'small'
        path = Path(cleaned)
        if path.exists():
            return str(path)
        from_url = self._extract_model_id_from_url(cleaned)
        if from_url:
            cleaned = from_url
        lowered = cleaned.lower()
        normalized = lowered.replace('_', '-')
        if lowered in self._MODEL_ALIAS:
            return self._MODEL_ALIAS[lowered]
        if normalized in self._MODEL_ALIAS:
            return self._MODEL_ALIAS[normalized]
        if '/' not in cleaned:
            prefixed = self._try_prefix_modelscope_owner(cleaned)
            if prefixed:
                return prefixed
        return cleaned

    @staticmethod
    def _extract_model_id_from_url(value: str) -> str | None:
        try:
            parsed = urlparse(value)
        except Exception:
            return None
        if parsed.scheme not in {'http', 'https'}:
            return None
        host = (parsed.netloc or '').lower()
        segments = [unquote(part) for part in (parsed.path or '').split('/') if part]
        if host.endswith('modelscope.cn'):
            if len(segments) >= 3 and segments[0].lower() == 'models':
                return f'{segments[1]}/{segments[2]}'
            return None
        if host.endswith('huggingface.co'):
            if len(segments) >= 2 and segments[0] not in {'models', 'spaces', 'datasets'}:
                return f'{segments[0]}/{segments[1]}'
            if len(segments) >= 3 and segments[0] in {'models', 'spaces', 'datasets'}:
                return f'{segments[1]}/{segments[2]}'
        return None

    @staticmethod
    def _try_prefix_modelscope_owner(model_name: str) -> str | None:
        lowered = model_name.strip().lower()
        if not lowered:
            return None
        if lowered.startswith('speech_paraformer'):
            return f'damo/{model_name.strip()}'
        if lowered.startswith('speech_') or lowered.startswith('sensevoice'):
            return f'iic/{model_name.strip()}'
        return None

    def _prepare_model_ref(self, model_ref: str, *, auto_download: bool, progress_callback) -> str | None:
        path = Path(model_ref)
        if path.exists():
            return str(path)
        if not auto_download or '/' not in model_ref:
            return None
        return self._download_modelscope_model(model_ref, progress_callback=progress_callback)

    def _build_model_with_progress(self, auto_model_factory: Any, model_kwargs: dict[str, object], *, auto_download: bool, progress_callback, model_name: str) -> Any:
        model_value = str(model_kwargs.get('model', '') or '').strip()
        if model_value and Path(model_value).exists():
            return _invoke_with_supported_kwargs(auto_model_factory, model_kwargs)
        if not auto_download or progress_callback is None:
            return _invoke_with_supported_kwargs(auto_model_factory, model_kwargs)
        emit_progress(progress_callback, f'Start initializing funasr model: {model_name}')
        stop_event = threading.Event()
        watch_roots = self._resolve_progress_watch_roots(model_name)
        watcher = threading.Thread(
            target=self._progress_monitor,
            args=(stop_event, progress_callback, model_name, watch_roots),
            daemon=True,
        )
        watcher.start()
        try:
            model = _invoke_with_supported_kwargs(auto_model_factory, model_kwargs)
            emit_progress(progress_callback, f'funasr model ready: {model_name}')
            return model
        finally:
            stop_event.set()
            watcher.join(timeout=1.5)

    def _download_modelscope_model(self, model_id: str, *, progress_callback) -> str | None:
        try:
            from modelscope import snapshot_download
        except Exception:
            emit_progress(progress_callback, 'funasr download skipped: modelscope is not available')
            return None

        target_root = library_model_dir('funasr')
        emit_progress(progress_callback, f'Start downloading funasr model: {model_id}')

        stop_event = threading.Event()
        watch_roots = self._resolve_progress_watch_roots(model_id, explicit_cache_root=target_root)
        watcher = threading.Thread(
            target=self._progress_monitor,
            args=(stop_event, progress_callback, model_id, watch_roots),
            daemon=True,
        )
        watcher.start()
        try:
            local_dir = snapshot_download(model_id, cache_dir=str(target_root))
            emit_progress(progress_callback, f'funasr model download completed: {model_id}')
            return str(local_dir)
        except Exception as exc:
            emit_progress(progress_callback, f'funasr model download failed: {exc}')
            return None
        finally:
            stop_event.set()
            watcher.join(timeout=1.5)

    @staticmethod
    def _progress_monitor(stop_event: threading.Event, progress_callback, model_name: str, watch_roots: list[Path]) -> None:
        start_ts = time.time()
        fallback_bytes = 0
        last_reported_bytes = -1
        while not stop_event.wait(0.8):
            observed_bytes = FunASRTranscriber._measure_recent_bytes(watch_roots, start_ts)
            if observed_bytes <= 0:
                fallback_bytes += 1024 * 1024
                display_bytes = fallback_bytes
            else:
                display_bytes = observed_bytes
            if display_bytes == last_reported_bytes:
                continue
            last_reported_bytes = display_bytes
            emit_progress(
                progress_callback,
                format_download_progress('funasr', model_name, display_bytes, None),
            )

    @staticmethod
    def _measure_recent_bytes(watch_roots: list[Path], start_ts: float) -> int:
        total_bytes = 0
        for root in watch_roots:
            if not root.exists():
                continue
            for item in root.rglob('*'):
                if not item.is_file():
                    continue
                try:
                    stat = item.stat()
                except Exception:
                    continue
                if stat.st_mtime + 2.0 < start_ts:
                    continue
                total_bytes += max(0, int(stat.st_size))
        return total_bytes

    @staticmethod
    def _resolve_progress_watch_roots(model_ref: str, explicit_cache_root: Path | None=None) -> list[Path]:
        bases: list[Path] = [library_model_dir('funasr')]
        if explicit_cache_root is not None:
            bases.append(explicit_cache_root)
        env_cache = os.environ.get('MODELSCOPE_CACHE', '').strip()
        if env_cache:
            bases.append(Path(env_cache))
        else:
            bases.append(Path.home() / '.cache' / 'modelscope')
        watch_roots: list[Path] = []
        model_id = model_ref.strip()
        has_owner_name = False
        if '/' in model_id:
            owner, name = model_id.split('/', 1)
            owner = owner.strip()
            name = name.strip()
            if owner and name:
                has_owner_name = True
                for base in bases:
                    watch_roots.append(base / 'hub' / owner / name)
                    watch_roots.append(base / 'hub' / 'models' / owner / name)
                    watch_roots.append(base / 'hub' / 'temp')
        if not has_owner_name:
            for base in bases:
                watch_roots.append(base / 'hub')
                watch_roots.append(base / 'hub' / 'models')
                watch_roots.append(base / 'hub' / 'temp')
        unique_roots: list[Path] = []
        seen: set[str] = set()
        for root in watch_roots:
            key = str(root).lower()
            if key in seen:
                continue
            seen.add(key)
            unique_roots.append(root)
        return unique_roots

    @staticmethod
    def _to_funasr_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        mapping = {'zh': 'zn', 'en': 'en', 'ja': 'ja', 'ko': 'ko'}
        return mapping.get(language, language)

    def _write_wav(self, file_path: Path, audio: np.ndarray) -> None:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._target_sample_rate)
            wav_file.writeframes(pcm.tobytes())

    @staticmethod
    def _extract_text(result: object) -> str:
        if result is None:
            return ''
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return str(result.get('text', '') or '').strip()
        if isinstance(result, list):
            chunks: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    value = str(item.get('text', '') or '').strip()
                else:
                    value = str(item).strip()
                if value:
                    chunks.append(value)
            return ' '.join(chunks).strip()
        return str(result).strip()

def _invoke_with_supported_kwargs(factory: Any, kwargs: dict[str, object]) -> Any:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    if any((param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())):
        return factory(**kwargs)
    filtered = {key: value for (key, value) in kwargs.items() if key in sig.parameters}
    return factory(**filtered)
