"""Offline NLLB CTranslate2 translation backend.

This adapter is intentionally lazy: constructor/import paths stay cheap, and
model download/load happens on a daemon warmup thread. Until ready, translate()
returns None so subtitles remain source-only instead of blocking the STT loop.
"""
from __future__ import annotations

import importlib
import importlib.util
import threading
from pathlib import Path
from typing import Callable, Optional

from ..model_paths import library_model_dir
from ..stt.model_download import download_hf_files_with_progress, emit_progress
from .base import TranslationState


DEFAULT_NLLB_MODEL_REPO = "facebook/nllb-200-distilled-600M"
DEFAULT_NLLB_MODEL_DIR = "nllb-200-distilled-600m-ct2-int8"
_MODEL_ALLOW_PATTERNS = [
    "config.json",
    "model.bin",
    "shared_vocabulary.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "*.model",
]

_SOURCE_FLORES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-hant": "zho_Hant",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
}


def _map_app_code_to_flores(code: str | None, *, target: bool = False) -> str | None:
    token = str(code or "").strip().lower()
    if target and token in {"zh-hant", "zh-tw", "zh-hk"}:
        return "zho_Hant"
    if target and token in {"zh", "zh-hans", "zh-cn", "zh-sg"}:
        return "zho_Hans"
    return _SOURCE_FLORES.get(token)


class NllbTranslator:
    def __init__(
        self,
        *,
        enabled: bool,
        source_code: str,
        target_code: str,
        model_dir: str | Path | None = None,
        model_repo: str = DEFAULT_NLLB_MODEL_REPO,
        auto_download: bool = True,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 4,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._requested_enabled = bool(enabled)
        self._enabled = False
        self._source_code = (source_code or "auto").strip().lower()
        self._target_code = (target_code or "zh").strip().lower()
        self._model_dir = Path(model_dir) if model_dir else library_model_dir("translation") / "nllb" / DEFAULT_NLLB_MODEL_DIR
        self._model_repo = str(model_repo or DEFAULT_NLLB_MODEL_REPO).strip() or DEFAULT_NLLB_MODEL_REPO
        self._auto_download = bool(auto_download)
        self._device = "cuda" if str(device or "").strip().lower() == "cuda" else "cpu"
        self._compute_type = (compute_type or "int8").strip().lower() or "int8"
        self._beam_size = max(1, int(beam_size or 4))
        self._on_status = on_status
        self._state = TranslationState(False, "Translation disabled.")
        self._translator = None
        self._tokenizer = None
        self._ready = False
        self._warmup_started = False

        if not self._requested_enabled:
            self._state = TranslationState(False, "Translation disabled by config.")
            return
        missing = self._missing_dependencies()
        if missing:
            self._state = TranslationState(False, f"NLLB backend unavailable: missing {', '.join(missing)}.")
            return
        target = _map_app_code_to_flores(self._target_code, target=True)
        if target is None:
            self._state = TranslationState(False, f"NLLB target language unsupported: {self._target_code}.")
            return
        self._state = TranslationState(False, "NLLB backend warming up.")
        self._start_warmup()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> "NllbTranslator":
        return cls(
            enabled=bool(getattr(config, "translation_enabled", False)),
            source_code=str(getattr(config, "translation_from", "auto") or "auto"),
            target_code=str(getattr(config, "translation_to", "zh") or "zh"),
            model_dir=str(getattr(config, "translation_nllb_model_path", "") or "").strip() or None,
            model_repo=str(getattr(config, "translation_nllb_model_repo", DEFAULT_NLLB_MODEL_REPO) or DEFAULT_NLLB_MODEL_REPO),
            auto_download=bool(getattr(config, "translation_nllb_auto_download", True)),
            device=str(getattr(config, "translation_nllb_device", "cpu") or "cpu"),
            compute_type=str(getattr(config, "translation_nllb_compute_type", "int8") or "int8"),
            on_status=on_status,
        )

    @property
    def name(self) -> str:
        return "nllb"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def state(self) -> TranslationState:
        return self._state

    def translate(self, text: str, source_code: str | None = None) -> Optional[str]:
        if not self._requested_enabled or not text.strip():
            return None
        source_flores = self._resolve_source_code(source_code)
        target_flores = _map_app_code_to_flores(self._target_code, target=True)
        if source_flores is None or target_flores is None or source_flores == target_flores:
            return None
        if not self._ready or self._translator is None or self._tokenizer is None:
            return None
        try:
            tokenizer = self._tokenizer
            translator = self._translator
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = source_flores
            source_ids = tokenizer.encode(text)
            source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
            result = translator.translate_batch(
                [source_tokens],
                target_prefix=[[target_flores]],
                beam_size=self._beam_size,
            )
            hypothesis = list(result[0].hypotheses[0])
            if hypothesis and hypothesis[0] == target_flores:
                hypothesis = hypothesis[1:]
            out_ids = tokenizer.convert_tokens_to_ids(hypothesis)
            translated = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            return translated or None
        except Exception:
            return None

    def _resolve_source_code(self, source_code: str | None) -> str | None:
        token = (source_code or "").strip().lower()
        if not token and self._source_code != "auto":
            token = self._source_code
        if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
            token = "zh-hant" if token in {"zh-hant", "zh-tw", "zh-hk"} else "zh"
        return _map_app_code_to_flores(token)

    @staticmethod
    def _missing_dependencies() -> list[str]:
        missing: list[str] = []
        for name in ("ctranslate2", "transformers"):
            try:
                if importlib.util.find_spec(name) is None:
                    missing.append(name)
            except Exception:
                missing.append(name)
        return missing

    def _start_warmup(self) -> None:
        if self._warmup_started:
            return
        self._warmup_started = True
        thread = threading.Thread(target=self._warmup, name="nllb-translation-warmup", daemon=True)
        thread.start()

    def _warmup(self) -> None:
        try:
            self._prepare_model_dir()
            if not self._is_ct2_model_ready(self._model_dir):
                self._state = TranslationState(
                    False,
                    f"NLLB model is not a ready CTranslate2 model: {self._model_dir}",
                )
                return
            ctranslate2 = importlib.import_module("ctranslate2")
            transformers = importlib.import_module("transformers")
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(self._model_dir), src_lang="eng_Latn")
            translator = ctranslate2.Translator(
                str(self._model_dir),
                device=self._device,
                compute_type=self._compute_type,
            )
            self._tokenizer = tokenizer
            self._translator = translator
            self._ready = True
            self._enabled = True
            self._state = TranslationState(
                True,
                f"NLLB translation active: target={self._target_code}; device={self._device}; compute={self._compute_type}",
            )
            self._emit(f"NLLB translation backend ready: {self._model_dir}")
        except Exception as exc:
            self._ready = False
            self._enabled = False
            self._state = TranslationState(False, f"NLLB backend unavailable: {exc}")

    def _prepare_model_dir(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if self._is_ct2_model_ready(self._model_dir):
            self._emit(f"[download] nllb cache hit: {self._model_dir}")
            return
        if not self._auto_download:
            return
        self._emit(f"[download] nllb preparing: {self._model_repo}")
        download_hf_files_with_progress(
            repo_id=self._model_repo,
            output_dir=str(self._model_dir),
            allow_patterns=_MODEL_ALLOW_PATTERNS,
            progress_callback=self._on_status,
            provider="nllb",
            model_name=self._model_repo,
        )

    @staticmethod
    def _is_ct2_model_ready(path: Path) -> bool:
        return bool((path / "config.json").exists() and (path / "model.bin").exists())

    def _emit(self, message: str) -> None:
        emit_progress(self._on_status, message)
