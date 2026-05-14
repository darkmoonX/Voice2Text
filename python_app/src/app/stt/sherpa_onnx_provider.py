"""Sherpa-ONNX provider adapter for transducer/paraformer offline recognition."""
from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from ..audio_capture import AudioChunk
from ..model_paths import library_model_dir
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample
from .model_assets import ensure_model_preset_downloaded

class SherpaOnnxTranscriber:
    """Sherpa-ONNX provider wrapper for transducer/paraformer local models."""

    def __init__(self, model_ref: str='small', provider: str='cpu', num_threads: int=2, target_sample_rate: int=16000, auto_download: bool=True, progress_callback: Callable[[str], None] | None=None) -> None:
        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError('sherpa-onnx is not installed. Run: pip install sherpa-onnx') from exc
        self._sherpa_onnx = sherpa_onnx
        self._target_sample_rate = int(target_sample_rate)
        self._provider = provider
        self._num_threads = max(1, int(num_threads))
        self._model_dir = self._resolve_model_dir(model_ref, auto_download=auto_download, progress_callback=progress_callback)
        self._active_paraformer_model: Path | None = None
        self._paraformer_fp32_model: Path | None = None
        self._paraformer_runtime_fallback_used = False
        self._recognizer = self._build_recognizer(sherpa_onnx=sherpa_onnx, model_dir=self._model_dir, provider=provider, num_threads=self._num_threads)

    def has_enough_signal(self, chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
        if not has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode):
            return False
        if self._active_paraformer_model is None:
            return True
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return False
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 5:
            return False
        return not self._looks_noise_like_for_paraformer(audio)

    def transcribe(self, chunk: AudioChunk, language: Optional[str]=None, channel_mode: str='mono') -> str:
        """Decode a chunk with Sherpa recognizer (language hint used for script normalization only)."""
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return ''
        audio = resample(audio, chunk.sample_rate, self._target_sample_rate)
        if audio.size < self._target_sample_rate // 5:
            return ''
        (_, zh_script) = normalize_language_hint(language)
        stream = self._create_stream_with_audio(audio)
        try:
            self._recognizer.decode_stream(stream)
        except Exception as exc:
            if self._try_recover_paraformer_decode_error(exc):
                stream = self._create_stream_with_audio(audio)
                try:
                    self._recognizer.decode_stream(stream)
                except Exception as retry_exc:
                    if self._is_paraformer_runtime_error(str(retry_exc)):
                        return ''
                    raise
            elif self._is_paraformer_runtime_error(str(exc)):
                return ''
            else:
                raise
        text = self._extract_text(getattr(stream, 'result', None))
        return normalize_chinese_script(text, zh_script)

    def _create_stream_with_audio(self, audio: Any) -> Any:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._target_sample_rate, audio)
        if hasattr(stream, 'input_finished'):
            try:
                stream.input_finished()
            except Exception:
                pass
        return stream

    @staticmethod
    def _looks_noise_like_for_paraformer(audio: np.ndarray) -> bool:
        if audio.size < 2048:
            return False
        window = audio
        if window.size > 65536:
            window = window[-65536:]
        zcr = float(np.mean(np.abs(np.diff(np.signbit(window)))))
        spectrum = np.abs(np.fft.rfft(window)) + 1e-10
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        return flatness >= 0.62 and zcr >= 0.04

    def _try_recover_paraformer_decode_error(self, exc: Exception) -> bool:
        if self._paraformer_runtime_fallback_used:
            return False
        active = self._active_paraformer_model
        fp32 = self._paraformer_fp32_model
        if active is None or fp32 is None:
            return False
        if active.resolve() == fp32.resolve():
            return False
        if not self._is_paraformer_runtime_error(str(exc)):
            return False
        try:
            self._recognizer = self._build_recognizer(sherpa_onnx=self._sherpa_onnx, model_dir=self._model_dir, provider=self._provider, num_threads=self._num_threads, force_paraformer_model=fp32)
            self._paraformer_runtime_fallback_used = True
            return True
        except Exception:
            return False

    @staticmethod
    def _is_paraformer_runtime_error(message: str) -> bool:
        lowered = (message or '').lower()
        return 'constantofshape' in lowered or 'tensor shape.size()' in lowered or 'onnxruntime' in lowered or ('loop_' in lowered) or ('decode streams' in lowered)

    def _resolve_model_dir(self, model_ref: str, auto_download: bool, progress_callback: Callable[[str], None] | None) -> Path:
        model_ref = model_ref.strip() or 'small'
        model_path = Path(model_ref)
        if model_path.exists():
            return model_path
        if '/' in model_ref or '\\' in model_ref:
            raise FileNotFoundError(f'Sherpa-ONNX model path not found: {model_ref}')
        model_root = library_model_dir('sherpa-onnx')
        candidate = model_root / model_ref
        if candidate.exists():
            return candidate
        if auto_download:
            downloaded = ensure_model_preset_downloaded(provider='sherpa-onnx', model_ref=model_ref, model_root=model_root, progress_callback=progress_callback)
            if downloaded is not None and downloaded.exists():
                return downloaded
        dirs = sorted([item for item in model_root.iterdir() if item.is_dir()])
        if len(dirs) == 1:
            return dirs[0]
        raise FileNotFoundError(f'Unable to locate Sherpa-ONNX model. Set --stt-model-path to a model folder containing encoder/decoder/joiner/tokens (transducer) or model(.int8).onnx + tokens (paraformer). Searched: {candidate}')

    def _build_recognizer(self, sherpa_onnx: Any, model_dir: Path, provider: str, num_threads: int, force_paraformer_model: Path | None=None) -> Any:
        (encoder, decoder, joiner) = self._find_transducer_model_files(model_dir)
        tokens = model_dir / 'tokens.txt'
        paraformer_model = force_paraformer_model or self._find_paraformer_model(model_dir)
        has_transducer = all((path is not None for path in (encoder, decoder, joiner))) and tokens.exists()
        has_paraformer = tokens.exists() and paraformer_model is not None
        if not has_transducer and (not has_paraformer):
            missing: list[str] = []
            if encoder is None:
                missing.append('encoder.onnx')
            if decoder is None:
                missing.append('decoder.onnx')
            if joiner is None:
                missing.append('joiner.onnx')
            raise FileNotFoundError('Sherpa-ONNX model directory is missing files: ' + ', '.join(missing))
        self._active_paraformer_model = paraformer_model
        if paraformer_model is not None:
            fp32 = model_dir / 'model.onnx'
            if fp32.exists():
                self._paraformer_fp32_model = fp32
            else:
                self._paraformer_fp32_model = None
        else:
            self._paraformer_fp32_model = None
        transducer_cfg_cls = getattr(sherpa_onnx, 'OfflineTransducerModelConfig', None)
        paraformer_cfg_cls = getattr(sherpa_onnx, 'OfflineParaformerModelConfig', None)
        model_cfg_cls = getattr(sherpa_onnx, 'OfflineModelConfig', None)
        recognizer_cfg_cls = getattr(sherpa_onnx, 'OfflineRecognizerConfig', None)
        recognizer_cls = getattr(sherpa_onnx, 'OfflineRecognizer', None)
        if not all([model_cfg_cls, recognizer_cfg_cls, recognizer_cls]):
            raise RuntimeError('Unsupported sherpa-onnx Python API. Expected OfflineRecognizer interfaces.')
        if has_transducer and hasattr(recognizer_cls, 'from_transducer'):
            try:
                assert encoder is not None
                assert decoder is not None
                assert joiner is not None
                return _invoke_with_supported_kwargs(recognizer_cls.from_transducer, {'encoder': str(encoder), 'decoder': str(decoder), 'joiner': str(joiner), 'tokens': str(tokens), 'num_threads': num_threads, 'sample_rate': self._target_sample_rate, 'feature_dim': 80, 'decoding_method': 'greedy_search', 'debug': False, 'provider': provider})
            except Exception:
                pass
        if has_paraformer and hasattr(recognizer_cls, 'from_paraformer'):
            try:
                assert paraformer_model is not None
                return _invoke_with_supported_kwargs(recognizer_cls.from_paraformer, {'paraformer': str(paraformer_model), 'tokens': str(tokens), 'num_threads': num_threads, 'sample_rate': self._target_sample_rate, 'feature_dim': 80, 'decoding_method': 'greedy_search', 'debug': False, 'provider': provider})
            except Exception:
                pass
        model_cfg = _new_config_instance(model_cfg_cls)
        _set_attr_if_present(model_cfg, 'tokens', str(tokens))
        _set_attr_if_present(model_cfg, 'num_threads', num_threads)
        _set_attr_if_present(model_cfg, 'provider', provider)
        _set_attr_if_present(model_cfg, 'debug', False)
        if has_transducer:
            if transducer_cfg_cls is None:
                raise RuntimeError('sherpa-onnx runtime does not expose OfflineTransducerModelConfig')
            assert encoder is not None
            assert decoder is not None
            assert joiner is not None
            transducer_cfg = self._build_transducer_config(transducer_cfg_cls, encoder=encoder, decoder=decoder, joiner=joiner)
            _set_attr_if_present(model_cfg, 'transducer', transducer_cfg)
        else:
            if paraformer_cfg_cls is None:
                raise RuntimeError('sherpa-onnx runtime does not expose OfflineParaformerModelConfig')
            assert paraformer_model is not None
            paraformer_cfg = self._build_paraformer_config(paraformer_cfg_cls, model_file=paraformer_model)
            _set_attr_if_present(model_cfg, 'paraformer', paraformer_cfg)
        recognizer_cfg = _new_config_instance(recognizer_cfg_cls)
        _set_attr_if_present(recognizer_cfg, 'model_config', model_cfg)
        _set_attr_if_present(recognizer_cfg, 'decoding_method', 'greedy_search')
        feature_cfg_cls = getattr(sherpa_onnx, 'FeatureConfig', None)
        if feature_cfg_cls is not None:
            feat_cfg = _new_config_instance(feature_cfg_cls)
            _set_attr_if_present(feat_cfg, 'sample_rate', self._target_sample_rate)
            _set_attr_if_present(feat_cfg, 'feature_dim', 80)
            _set_attr_if_present(recognizer_cfg, 'feat_config', feat_cfg)
        try:
            return recognizer_cls(recognizer_cfg)
        except TypeError:
            return recognizer_cls(config=recognizer_cfg)

    @staticmethod
    def _find_paraformer_model(model_dir: Path) -> Path | None:
        fp_model = model_dir / 'model.onnx'
        if fp_model.exists():
            return fp_model
        int8_model = model_dir / 'model.int8.onnx'
        if int8_model.exists():
            return int8_model
        return None

    @staticmethod
    def _find_transducer_model_files(model_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
        return (SherpaOnnxTranscriber._find_model_file(model_dir, 'encoder'), SherpaOnnxTranscriber._find_model_file(model_dir, 'decoder'), SherpaOnnxTranscriber._find_model_file(model_dir, 'joiner'))

    @staticmethod
    def _find_model_file(model_dir: Path, prefix: str) -> Path | None:
        exact = model_dir / f'{prefix}.onnx'
        if exact.exists():
            return exact
        non_int8 = sorted((path for path in model_dir.glob(f'{prefix}-*.onnx') if not path.name.endswith('.int8.onnx')))
        if non_int8:
            return non_int8[0]
        int8 = sorted(model_dir.glob(f'{prefix}-*.int8.onnx'))
        if int8:
            return int8[0]
        return None

    @staticmethod
    def _build_transducer_config(transducer_cfg_cls: Any, *, encoder: Path, decoder: Path, joiner: Path) -> Any:
        try:
            return transducer_cfg_cls(encoder_filename=str(encoder), decoder_filename=str(decoder), joiner_filename=str(joiner))
        except Exception:
            pass
        try:
            return transducer_cfg_cls(str(encoder), str(decoder), str(joiner))
        except Exception:
            pass
        return _invoke_with_supported_kwargs(transducer_cfg_cls, {'encoder': str(encoder), 'decoder': str(decoder), 'joiner': str(joiner), 'encoder_filename': str(encoder), 'decoder_filename': str(decoder), 'joiner_filename': str(joiner)})

    @staticmethod
    def _build_paraformer_config(paraformer_cfg_cls: Any, *, model_file: Path) -> Any:
        try:
            return paraformer_cfg_cls(model=str(model_file))
        except Exception:
            pass
        try:
            cfg = paraformer_cfg_cls()
        except Exception:
            cfg = _invoke_with_supported_kwargs(paraformer_cfg_cls, {})
        _set_attr_if_present(cfg, 'model', str(model_file))
        return cfg

    @staticmethod
    def _extract_text(result: object) -> str:
        if result is None:
            return ''
        if isinstance(result, str):
            return result.strip()
        text = getattr(result, 'text', None)
        if isinstance(text, str):
            return text.strip()
        if isinstance(result, dict):
            value = result.get('text', '')
            return str(value).strip()
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

def _new_config_instance(factory: Any) -> Any:
    try:
        return factory()
    except Exception:
        return _invoke_with_supported_kwargs(factory, {})

def _set_attr_if_present(target: Any, name: str, value: Any) -> None:
    if not hasattr(target, name):
        return
    try:
        setattr(target, name, value)
    except Exception:
        return
