"""Local-LLM translation backend via a resident llama.cpp server (round 0074).

Fills the `llm` registry slot reserved in round 0026. Runs `llama-server` as a managed
subprocess (mirroring the whisper.cpp resident-server pattern) and translates one
subtitle line per OpenAI-compatible `/v1/chat/completions` call at temperature 0.

Spike results (round 0073, Qwen3-4B-Instruct-2507 Q4_K_M on an RTX 3060): clearly better
than Argos in the primary en->zh-hant direction, p95 0.34 s/line, coexists with the
large-v3 ASR model in VRAM. The system prompt carries a "keep numbers/units verbatim"
guard for the number-semantics slip class observed in the spike (4到5個月 -> "April to
May" at temperature 0).

Like `NllbTranslator`, this adapter is lazy: the constructor stays cheap, server spawn +
health polling happen on a daemon warmup thread, and `translate()` returns None until
ready (subtitles stay source-only; the no-translation fallback is never broken). If a
healthy server is already listening on the configured port it is reused and NOT owned
(nothing is killed on shutdown).
"""
from __future__ import annotations

import atexit
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .base import TranslationState

DEFAULT_LLM_PORT = 8474

_TARGET_LABELS = {
    "zh": "Simplified Chinese (简体中文)",
    "zh-hans": "Simplified Chinese (简体中文)",
    "zh-cn": "Simplified Chinese (简体中文)",
    "zh-sg": "Simplified Chinese (简体中文)",
    "zh-hant": "Traditional Chinese (繁體中文)",
    "zh-tw": "Traditional Chinese (繁體中文)",
    "zh-hk": "Traditional Chinese (繁體中文)",
    "en": "natural English",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

_ZH_FAMILY = {"zh", "zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk", "zh-sg"}


def _norm(code: str | None) -> str:
    return str(code or "").strip().lower()


def _language_family(code: str | None) -> str:
    token = _norm(code)
    return "zh" if token in _ZH_FAMILY else token


def target_label(target_code: str | None) -> str:
    token = _norm(target_code)
    return _TARGET_LABELS.get(token, f"the language with code '{token}'")


def build_system_prompt(target_code: str | None) -> str:
    return (
        "You are a professional subtitle translator. Translate the user's subtitle line "
        f"into {target_label(target_code)}. Keep numbers, units, and proper nouns exactly "
        "as in the source. Output ONLY the translation - no explanations, no quotes, no "
        "romanization."
    )


class LlmTranslator:
    def __init__(
        self,
        *,
        enabled: bool,
        source_code: str,
        target_code: str,
        server_path: str = "",
        model_path: str = "",
        port: int = DEFAULT_LLM_PORT,
        context_size: int = 4096,
        gpu_layers: int = 99,
        max_output_tokens: int = 256,
        request_timeout_seconds: float = 10.0,
        health_timeout_seconds: float = 120.0,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._requested_enabled = bool(enabled)
        self._source_code = _norm(source_code) or "auto"
        self._target_code = _norm(target_code) or "zh"
        self._server_path = str(server_path or "").strip()
        self._model_path = str(model_path or "").strip()
        self._port = int(port or DEFAULT_LLM_PORT)
        self._context_size = max(512, int(context_size or 4096))
        self._gpu_layers = max(0, int(gpu_layers or 0))
        self._max_output_tokens = max(16, int(max_output_tokens or 256))
        self._request_timeout = max(1.0, float(request_timeout_seconds or 10.0))
        self._health_timeout = max(5.0, float(health_timeout_seconds or 120.0))
        self._on_status = on_status
        self._state = TranslationState(False, "Translation disabled.")
        self._ready = False
        self._proc: subprocess.Popen | None = None
        self._owns_server = False
        self._warmup_started = False
        self._shutdown_registered = False

        if not self._requested_enabled:
            self._state = TranslationState(False, "Translation disabled by config.")
            return
        if not self._server_path or not Path(self._server_path).exists():
            self._state = TranslationState(
                False,
                "LLM backend unavailable: translation_llm_server_path is not set or does "
                f"not exist ({self._server_path or 'empty'}).",
            )
            return
        if not self._model_path or not Path(self._model_path).exists():
            self._state = TranslationState(
                False,
                "LLM backend unavailable: translation_llm_model_path is not set or does "
                f"not exist ({self._model_path or 'empty'}).",
            )
            return
        self._state = TranslationState(False, "LLM translation backend warming up.")
        self._start_warmup()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> "LlmTranslator":
        return cls(
            enabled=bool(getattr(config, "translation_enabled", False)),
            source_code=str(getattr(config, "translation_from", "auto") or "auto"),
            target_code=str(getattr(config, "translation_to", "zh") or "zh"),
            server_path=str(getattr(config, "translation_llm_server_path", "") or ""),
            model_path=str(getattr(config, "translation_llm_model_path", "") or ""),
            port=int(getattr(config, "translation_llm_port", DEFAULT_LLM_PORT) or DEFAULT_LLM_PORT),
            context_size=int(getattr(config, "translation_llm_context_size", 4096) or 4096),
            gpu_layers=int(getattr(config, "translation_llm_gpu_layers", 99) or 0),
            max_output_tokens=int(getattr(config, "translation_llm_max_output_tokens", 256) or 256),
            request_timeout_seconds=float(getattr(config, "translation_llm_request_timeout_seconds", 10.0) or 10.0),
            on_status=on_status,
        )

    # --- TranslationBackend protocol ---

    @property
    def name(self) -> str:
        return "llm"

    @property
    def enabled(self) -> bool:
        return self._ready

    @property
    def state(self) -> TranslationState:
        return self._state

    def translate(self, text: str, source_code: str | None = None) -> Optional[str]:
        if not self._requested_enabled or not self._ready:
            return None
        line = str(text or "").strip()
        if not line:
            return None
        source = _norm(source_code) or (self._source_code if self._source_code != "auto" else "")
        if source and _language_family(source) == _language_family(self._target_code):
            return None
        try:
            body = json.dumps({
                "messages": [
                    {"role": "system", "content": build_system_prompt(self._target_code)},
                    {"role": "user", "content": line},
                ],
                "temperature": 0.0,
                "max_tokens": self._max_output_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{self._port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            translated = str(payload["choices"][0]["message"]["content"] or "").strip()
            return translated or None
        except Exception:
            return None

    # --- lifecycle ---

    def shutdown(self) -> None:
        """Kill the owned server subprocess (no-op for a reused external server)."""
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is not None and self._owns_server:
            try:
                proc.kill()
            except Exception:
                pass

    def _start_warmup(self) -> None:
        if self._warmup_started:
            return
        self._warmup_started = True
        thread = threading.Thread(target=self._warmup, name="llm-translation-warmup", daemon=True)
        thread.start()

    def _warmup(self) -> None:
        try:
            if self._health_ok(timeout=2.0):
                self._owns_server = False
                self._emit(f"LLM translation: reusing existing llama-server on port {self._port}.")
            else:
                self._spawn_server()
                if not self._wait_healthy():
                    self.shutdown()
                    self._state = TranslationState(
                        False,
                        f"LLM backend unavailable: llama-server did not become healthy within "
                        f"{self._health_timeout:.0f}s (port {self._port}).",
                    )
                    return
            self._ready = True
            self._state = TranslationState(
                True,
                f"LLM translation active: target={self._target_code}; "
                f"model={Path(self._model_path).name}; port={self._port}",
            )
            self._emit(f"LLM translation backend ready: {Path(self._model_path).name}")
        except Exception as exc:
            self._ready = False
            self._state = TranslationState(False, f"LLM backend unavailable: {exc}")

    def _spawn_server(self) -> None:
        args = [
            self._server_path,
            "-m", self._model_path,
            "-ngl", str(self._gpu_layers),
            "--port", str(self._port),
            "-c", str(self._context_size),
            "--no-webui",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._owns_server = True
        if not self._shutdown_registered:
            self._shutdown_registered = True
            atexit.register(self.shutdown)
        self._emit(f"LLM translation: started llama-server (pid={self._proc.pid}, port={self._port}).")

    def _wait_healthy(self) -> bool:
        import time
        deadline = time.monotonic() + self._health_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # server process died during startup
            if self._health_ok(timeout=2.0):
                return True
            time.sleep(1.0)
        return False

    def _health_ok(self, *, timeout: float) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def _emit(self, message: str) -> None:
        if self._on_status:
            try:
                self._on_status(message)
            except Exception:
                pass
