"""WhisperX provider adapter with optional alignment for subtitle quality."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional
import threading
import time
import warnings

from ..model_paths import library_model_dir
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample
from .model_download import download_hf_snapshot_with_progress, emit_progress, format_download_progress
import re


class WhisperXTranscriber:
    def __init__(
        self,
        model_ref: str = "small",
        *,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 4,
        enable_phoneme_asr: bool = True,
        enable_forced_alignment: bool = True,
        enable_vad: bool = True,
        vad_method: str = "silero-vad",
        enable_diarization: bool = False,
        alignment_model: str = "",
        alignment_language: str = "auto",
        source_language_hint: str | None = None,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
        auto_download: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        warnings.filterwarnings(
            'ignore',
            message=r'.*TensorFloat-32 \(TF32\) has been disabled.*',
            module=r'pyannote\.audio\.utils\.reproducibility',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'.*torchcodec is not installed correctly.*',
            module=r'pyannote\.audio\.core\.io',
        )
        try:
            import whisperx  # type: ignore
        except Exception as exc:  # pragma: no cover - import path depends on env
            raise RuntimeError(f"whisperx is not available: {exc}") from exc

        self._whisperx = whisperx
        self._model_ref = (model_ref or "small").strip() or "small"
        self._device = "cpu" if device.strip().lower().startswith("cpu") else "cuda"
        self._compute_type = (compute_type or "float16").strip()
        self._batch_size = max(1, int(batch_size))
        self._enable_phoneme_asr = bool(enable_phoneme_asr)
        self._enable_forced_alignment = bool(enable_forced_alignment)
        self._enable_vad = bool(enable_vad)
        self._vad_method = (vad_method or "silero-vad").strip().lower()
        self._enable_diarization = bool(enable_diarization)
        self._alignment_model = (alignment_model or "").strip()
        self._alignment_language = (alignment_language or 'auto').strip().lower()
        (normalized_source_lang, _) = normalize_language_hint(source_language_hint)
        self._source_language_hint = normalized_source_lang
        self._diarization_model = (diarization_model or "pyannote/speaker-diarization-3.1").strip()
        self._hf_token = (hf_token or "").strip()
        self._auto_download = bool(auto_download)
        self._progress_callback = progress_callback

        self._model_root = library_model_dir("whisperx")
        self._download_probe_roots = self._build_download_probe_roots()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        model_arg = self._resolve_stt_model_arg(self._model_ref)
        self._model = whisperx.load_model(
            model_arg,
            self._device,
            compute_type=self._compute_type,
            language=self._source_language_hint,
            vad_method=self._resolve_whisperx_vad_method(self._vad_method),
            download_root=str(self._model_root / "stt"),
        )

        self._align_cache: dict[str, tuple[object, object]] = {}
        self._diarization_pipeline = None
        self._last_transcription_meta: dict[str, object] = {}
        self._language_route_logged: set[tuple[str, str, str, str]] = set()
        self._emit(f"WhisperX initialized: model={model_arg}, device={self._device}")

    def has_enough_signal(self, chunk, threshold: float = 0.008, channel_mode: str = "mono") -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk, language: Optional[str] = None, channel_mode: str = "mono") -> str:
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        if audio.size == 0:
            return ""
        if chunk.sample_rate != 16000:
            audio = resample(audio, chunk.sample_rate, 16000)
        if audio.size == 0:
            return ""

        (lang_hint, zh_script) = normalize_language_hint(language)
        kwargs: dict[str, object] = {"batch_size": self._batch_size}
        if lang_hint is not None:
            kwargs["language"] = lang_hint
        if not self._enable_phoneme_asr:
            kwargs["task"] = "transcribe"

        result = self._transcribe_with_compat(audio, kwargs)
        segments = list(result.get("segments", []) or [])
        lang_detected = str(result.get("language") or "").strip().lower()
        align_lang = self._resolve_alignment_language(lang_hint, lang_detected)
        self._emit_language_route(language, lang_hint, lang_detected, align_lang)
        aligned = self._align_segments(audio, segments, align_lang) if self._enable_forced_alignment else segments
        if self._enable_diarization:
            aligned = self._attach_speaker_labels(audio, aligned)

        token_meta: list[dict[str, float]] = []
        words_fallback: list[str] = []
        text = " ".join((str(seg.get("text") or "").strip() for seg in aligned if str(seg.get("text") or "").strip())).strip()
        for seg in aligned:
            for wd in (seg.get("words") or []):
                try:
                    start = float(wd.get("start"))
                    end = float(wd.get("end"))
                    score = float(wd.get("score")) if wd.get("score") is not None else 0.0
                    word_txt = str(wd.get("word") or "").strip()
                    token_meta.append({"start": start, "end": end, "score": score, "word": word_txt})
                    if word_txt:
                        words_fallback.append(word_txt)
                except Exception:
                    continue
        if not text and words_fallback:
            text = " ".join(words_fallback).strip()
        if not text:
            text = str(result.get("text") or "").strip()
        stable = [tok for tok in token_meta if tok.get("score", 0.0) >= 0.60 and 0.02 <= (tok.get("end", 0.0) - tok.get("start", 0.0)) <= 1.2]
        self._last_transcription_meta = {
            "stability_ratio": float(len(stable) / max(1, len(token_meta))) if token_meta else 1.0,
            "token_count": int(len(token_meta)),
            "stable_token_count": int(len(stable)),
            "token_timestamps": token_meta[:80],
        }
        if zh_script is not None:
            text = normalize_chinese_script(text, zh_script)
        if not text:
            self._emit("WhisperX produced empty text after alignment/postprocess.")
        return text


    def _transcribe_with_compat(self, audio, kwargs: dict[str, object]):
        active = dict(kwargs)
        for _ in range(6):
            try:
                return self._model.transcribe(audio, **active)
            except TypeError as exc:
                msg = str(exc)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad = m.group(1)
                if bad in active:
                    self._emit(f"WhisperX compatibility: dropping unsupported kwarg '{bad}'")
                    active.pop(bad, None)
                    continue
                raise
        return self._model.transcribe(audio)

    def _resolve_alignment_language(self, lang_hint: Optional[str], lang_detected: str) -> str:
        mode = (self._alignment_language or 'auto').strip().lower()

        def _norm(token: str) -> str:
            value = (token or '').strip().lower()
            if value in {'zh-hant', 'zh-hans'}:
                return 'zh'
            return value

        if mode == 'follow-source':
            return _norm(lang_hint or self._source_language_hint or '')
        if mode not in {'', 'auto'}:
            return _norm(mode)
        return _norm(lang_hint or self._source_language_hint or lang_detected or '')


    @staticmethod
    def _resolve_whisperx_vad_method(method: str) -> str:
        token = (method or 'silero-vad').strip().lower()
        if token in {'silero', 'silero-vad'}:
            return 'silero'
        return 'pyannote'

    def _emit_language_route(self, source_language: Optional[str], lang_hint: Optional[str], lang_detected: str, align_lang: str) -> None:
        mode = (self._alignment_language or 'auto').strip().lower()
        key = (
            (source_language or 'auto').strip().lower(),
            (lang_hint or '').strip().lower(),
            (lang_detected or '').strip().lower(),
            (align_lang or '').strip().lower(),
        )
        if key in self._language_route_logged:
            return
        self._language_route_logged.add(key)
        self._emit(
            "WhisperX language routing: "
            f"source={key[0] or 'auto'}; asr_hint={key[1] or 'auto'}; "
            f"detected={key[2] or 'n/a'}; align_mode={mode}; align={key[3] or 'n/a'}"
        )
    def _align_segments(self, audio, segments: list[dict], language_code: str) -> list[dict]:
        if not segments or not language_code:
            return segments
        try:
            align_repo_id = self._resolve_alignment_repo_id(language_code)
            align_local_dir = self._resolve_alignment_local_dir(language_code, align_repo_id)
            explicit_model = self._alignment_model.strip()
            model_selection = explicit_model if explicit_model else f"auto:{align_repo_id or language_code}"
            cache_key = f"{language_code}|{model_selection.lower()}"

            if cache_key not in self._align_cache:
                self._emit(
                    f"WhisperX alignment model loading: language={language_code}; "
                    f"model={model_selection}; repo={(align_repo_id or 'auto-map')}; cache_key={cache_key}"
                )
                self._prepare_optional_hf_repo_download(
                    repo_id=align_repo_id,
                    local_dir=align_local_dir,
                    provider="whisperx-align",
                    model_name=(explicit_model or language_code),
                )
                kwargs: dict[str, object] = {
                    "language_code": language_code,
                    "device": self._device,
                    "model_dir": str(self._model_root / "align"),
                }
                local_repo_ready = bool(align_repo_id) and align_local_dir.exists() and any(align_local_dir.rglob('*'))
                if local_repo_ready:
                    kwargs["model_name"] = str(align_local_dir)
                    model_selection = str(align_local_dir)
                elif explicit_model:
                    kwargs["model_name"] = explicit_model
                model_a, metadata = self._run_with_external_download_progress(
                    f"align-{language_code}",
                    lambda: self._whisperx.load_align_model(**kwargs),
                )
                self._align_cache[cache_key] = (model_a, metadata)
                self._emit(
                    f"WhisperX alignment model ready: language={language_code}; "
                    f"model={model_selection}; cache_key={cache_key}"
                )

            (align_model, metadata) = self._align_cache[cache_key]
            aligned_result = self._whisperx.align(segments, align_model, metadata, audio, self._device, return_char_alignments=False)
            return list(aligned_result.get("segments", segments))
        except Exception as exc:
            self._emit(f"WhisperX alignment skipped: {exc}")
            return segments

    def _attach_speaker_labels(self, audio, segments: list[dict]) -> list[dict]:
        if not self._enable_diarization:
            return segments
        try:
            if self._diarization_pipeline is None:
                self._emit("WhisperX diarization model loading...")
                self._prepare_optional_hf_repo_download(
                    repo_id=self._diarization_model if "/" in self._diarization_model else "",
                    local_dir=self._model_root / "diarization",
                    provider="whisperx-diarization",
                    model_name=self._diarization_model,
                )
                kwargs: dict[str, object] = {"use_auth_token": self._hf_token or None, "device": self._device}
                if self._diarization_model:
                    kwargs["model_name"] = self._diarization_model
                self._diarization_pipeline = self._run_with_external_download_progress("diarization", lambda: self._whisperx.DiarizationPipeline(**kwargs))
                self._emit("WhisperX diarization pipeline initialized.")

            diarize_segments = self._diarization_pipeline(audio)
            aligned = self._whisperx.assign_word_speakers(diarize_segments, {"segments": segments})
            return list((aligned.get("segments", segments) if isinstance(aligned, dict) else segments))
        except Exception as exc:
            self._emit(f"WhisperX diarization skipped: {exc}")
            return segments

    def _resolve_stt_model_arg(self, model_ref: str) -> str:
        if not model_ref:
            return "small"
        if "/" in model_ref or "\\" in model_ref:
            return model_ref

        target_dir = self._model_root / "stt" / model_ref
        if (target_dir / "model.bin").exists() and (target_dir / "config.json").exists():
            return str(target_dir)

        repo_id = self._resolve_whisper_repo_id(model_ref)
        if repo_id and self._auto_download:
            self._emit(f"WhisperX STT model download started: {model_ref}")
            download_hf_snapshot_with_progress(
                repo_id=repo_id,
                output_dir=str(target_dir),
                allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
                progress_callback=self._progress_callback,
                provider="whisperx",
                model_name=model_ref,
            )
            self._emit(f"WhisperX STT model download completed: {model_ref}")
            return str(target_dir)

        return model_ref

    def _prepare_optional_hf_repo_download(self, *, repo_id: str, local_dir: Path, provider: str, model_name: str) -> None:
        if not repo_id or (not self._auto_download):
            return
        try:
            download_hf_snapshot_with_progress(
                repo_id=repo_id,
                output_dir=str(local_dir),
                allow_patterns=["*.bin", "*.json", "*.safetensors", "*.model", "*.txt", "*.yaml", "*.pt", "*.ckpt", "*.index", "*.onnx"],
                progress_callback=self._progress_callback,
                provider=provider,
                model_name=model_name,
                token=(self._hf_token or None),
            )
        except Exception as exc:
            self._emit(f"WhisperX optional pre-download skipped: {exc}")

    def _resolve_alignment_repo_id(self, language_code: str) -> str:
        # If user explicitly provided an HF repo id, use it directly.
        if self._alignment_model and ('/' in self._alignment_model):
            return self._alignment_model
        # If user provided a local/custom model name without repo id, skip predownload.
        if self._alignment_model:
            return ''
        try:
            alignment_mod = getattr(self._whisperx, 'alignment', None)
            if alignment_mod is None:
                return ''
            mapping_hf = getattr(alignment_mod, 'DEFAULT_ALIGN_MODELS_HF', None)
            if isinstance(mapping_hf, dict):
                repo = mapping_hf.get(language_code)
                if isinstance(repo, str) and repo.strip():
                    return repo.strip()
        except Exception:
            return ''
        return ''

    @staticmethod
    def _slugify_repo_id(value: str) -> str:
        token = (value or '').strip().lower()
        if not token:
            return 'auto'
        slug = re.sub(r'[^0-9a-z._-]+', '-', token)
        slug = slug.strip('-')
        return slug or 'auto'

    def _resolve_alignment_local_dir(self, language_code: str, repo_id: str) -> Path:
        base = self._model_root / 'align' / (language_code or 'auto')
        if repo_id:
            return base / self._slugify_repo_id(repo_id)
        if self._alignment_model:
            return base / 'custom'
        return base / 'auto'
    @staticmethod
    def _resolve_whisper_repo_id(model_ref: str) -> str | None:
        value = model_ref.strip()
        if not value:
            return None
        if "/" in value:
            return value
        try:
            from faster_whisper import utils as fw_utils  # type: ignore
            mapping = getattr(fw_utils, "_MODELS", None)
            if isinstance(mapping, dict):
                repo = mapping.get(value)
                if isinstance(repo, str) and repo.strip():
                    return repo
        except Exception:
            return None
        return None

    def _build_download_probe_roots(self) -> list[Path]:
        roots: list[Path] = []
        candidates = [
            os.environ.get("HF_HOME", ""),
            os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
            os.environ.get("TRANSFORMERS_CACHE", ""),
            str(Path.home() / ".cache" / "huggingface"),
            str(Path.home() / ".cache" / "torch"),
            str(self._model_root),
        ]
        for raw in candidates:
            if not raw:
                continue
            try:
                path = Path(raw).expanduser().resolve()
            except Exception:
                continue
            if path not in roots:
                roots.append(path)
        return roots

    @staticmethod
    def _safe_dir_size(root: Path) -> int:
        total = 0
        try:
            if not root.exists():
                return 0
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    total += int(path.stat().st_size)
                except Exception:
                    continue
        except Exception:
            return total
        return total

    def _current_probe_size(self) -> int:
        return sum(self._safe_dir_size(root) for root in self._download_probe_roots)

    def _run_with_external_download_progress(self, label: str, fn):
        if self._progress_callback is None:
            return fn()

        done = threading.Event()
        before_size = self._current_probe_size()
        state = {"peak_delta": 0}

        def monitor() -> None:
            while not done.is_set():
                try:
                    now_size = self._current_probe_size()
                    delta = max(0, now_size - before_size)
                    if delta > state["peak_delta"]:
                        state["peak_delta"] = delta
                    if delta <= 0 and state["peak_delta"] <= 0:
                        done.wait(0.6)
                        continue
                    total_for_display = max(state["peak_delta"], delta, 1)
                    emit_progress(
                        self._progress_callback,
                        format_download_progress("download", f"whisperx:{label}", delta, total_for_display),
                    )
                except Exception as exc:
                    emit_progress(self._progress_callback, f"download monitor skipped: {exc}")
                    break
                done.wait(0.6)

        thread = threading.Thread(target=monitor, name=f"whisperx-dl-{label}", daemon=True)
        thread.start()
        try:
            return fn()
        finally:
            done.set()
            thread.join(timeout=1.0)
            after_size = self._current_probe_size()
            final_delta = max(0, after_size - before_size)
            total = max(state["peak_delta"], final_delta, 0)
            if total > 0 or final_delta > 0:
                try:
                    emit_progress(
                        self._progress_callback,
                        format_download_progress("download", f"whisperx:{label}", final_delta, max(1, total)),
                    )
                except Exception as exc:
                    emit_progress(self._progress_callback, f"download final-progress skipped: {exc}")
    def _emit(self, message: str) -> None:
        emit_progress(self._progress_callback, message)

    def get_last_transcription_meta(self) -> dict[str, object]:
        return dict(self._last_transcription_meta)







