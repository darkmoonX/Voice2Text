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

# --- VRAM-aware auto sizing (round 0075) ---------------------------------------
# translation_llm_context_size <= 0 means AUTO: probe free VRAM at warmup, subtract a
# conservative reserve for the (possibly not-yet-loaded) ASR stack, and pick the largest
# context tier that fits — dropping to CPU inference when even the smallest doesn't.
# An explicit positive value is a manual pin: auto sizing never touches it (startup/
# runtime recovery may still lower GPU layers, but the context stays as pinned).

AUTO_CONTEXT_SIZE = 0
CONTEXT_TIERS = (4096, 2048, 1024)
_CONSERVATIVE_AUTO_CONTEXT = 2048   # used when free-VRAM probing is unavailable
_KV_MIB_PER_1K_CTX = 150            # rough KV+compute cost per 1024 ctx for a 4B-class model
_RUNTIME_MARGIN_MIB = 600           # llama.cpp runtime buffers beyond weights+KV


def probe_free_vram_mib() -> int | None:
    """Free VRAM in MiB via nvidia-smi (None when unavailable — non-NVIDIA/no driver)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        values = [int(v.strip()) for v in out.stdout.splitlines() if v.strip().isdigit()]
        return max(values) if values else None
    except Exception:
        return None


def estimate_asr_reserve_mib(config: object) -> int:
    """Conservative VRAM reserve for the ASR stack described by `config`.

    The LLM warmup thread races the STT warmup, so we cannot know whether the ASR model
    is already resident (in which case free VRAM would already exclude it). Always
    reserving is the safe bias: worst case the auto tier is one step smaller than
    strictly needed; it never OOMs the ASR load that comes after us.
    """
    provider = str(getattr(config, "stt_provider", "whisperx") or "whisperx").strip().lower()
    if provider == "whispercpp":
        size = str(getattr(config, "stt_whispercpp_model_size", "medium") or "medium").lower()
        reserve = 3400 if "large" in size else 1800
    else:
        device = str(getattr(config, "model_device", "cuda") or "cuda")
        if not device.lower().startswith("cuda"):
            return 0
        try:
            from ..stt.model_resolution import resolve_model_size
            size = resolve_model_size(getattr(config, "model_size", "auto"), device).lower()
        except Exception:
            size = "large-v3"
        if "large" in size or "turbo" in size:
            reserve = 4200
        elif "medium" in size:
            reserve = 2600
        else:
            reserve = 1800
        reserve += 600  # torch CUDA context
    if bool(getattr(config, "whisperx_enable_diarization", False)):
        reserve += 1300
    return reserve


def llm_footprint_mib(weights_mib: int, context_size: int) -> int:
    return int(weights_mib) + (context_size // 1024) * _KV_MIB_PER_1K_CTX + _RUNTIME_MARGIN_MIB


def choose_context_tier(budget_mib: int, weights_mib: int) -> tuple[int, bool]:
    """Largest context tier whose footprint fits `budget_mib`; (tier, use_gpu).

    use_gpu=False means not even the smallest tier fits — run the LLM on CPU (slow but
    functional; the engine queue policy is the escape hatch for pacing).
    """
    for ctx in CONTEXT_TIERS:
        if budget_mib >= llm_footprint_mib(weights_mib, ctx):
            return ctx, True
    return CONTEXT_TIERS[-1], False

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
        context_size: int = AUTO_CONTEXT_SIZE,
        gpu_layers: int = 99,
        max_output_tokens: int = 256,
        request_timeout_seconds: float = 10.0,
        health_timeout_seconds: float = 120.0,
        asr_reserve_mib: int = 0,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._requested_enabled = bool(enabled)
        self._source_code = _norm(source_code) or "auto"
        self._target_code = _norm(target_code) or "zh"
        self._server_path = str(server_path or "").strip()
        self._model_path = str(model_path or "").strip()
        self._port = int(port or DEFAULT_LLM_PORT)
        requested_context = int(context_size or 0)
        self._context_auto = requested_context <= 0
        self._context_size = 0 if self._context_auto else max(512, requested_context)
        self._gpu_layers = max(0, int(gpu_layers or 0))
        self._max_output_tokens = max(16, int(max_output_tokens or 256))
        self._request_timeout = max(1.0, float(request_timeout_seconds or 10.0))
        self._health_timeout = max(5.0, float(health_timeout_seconds or 120.0))
        self._asr_reserve_mib = max(0, int(asr_reserve_mib or 0))
        self._on_status = on_status
        self._state = TranslationState(False, "Translation disabled.")
        self._ready = False
        self._proc: subprocess.Popen | None = None
        self._owns_server = False
        self._warmup_started = False
        self._shutdown_registered = False
        self._heal_attempted = False
        # Effective launch parameters; resolved by auto sizing / retry ladder at warmup.
        self._effective_context = self._context_size or _CONSERVATIVE_AUTO_CONTEXT
        self._effective_gpu_layers = self._gpu_layers

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
            context_size=int(getattr(config, "translation_llm_context_size", AUTO_CONTEXT_SIZE) or 0),
            gpu_layers=int(getattr(config, "translation_llm_gpu_layers", 99) or 0),
            max_output_tokens=int(getattr(config, "translation_llm_max_output_tokens", 256) or 256),
            request_timeout_seconds=float(getattr(config, "translation_llm_request_timeout_seconds", 10.0) or 10.0),
            asr_reserve_mib=estimate_asr_reserve_mib(config),
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
            self._maybe_self_heal()
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
                self._resolve_auto_sizing()
                if not self._start_with_ladder():
                    self._state = TranslationState(
                        False,
                        "LLM backend unavailable: llama-server failed to start (all retry "
                        f"tiers exhausted, port {self._port}).",
                    )
                    return
            self._ready = True
            self._state = TranslationState(True, self._active_message())
            self._emit(f"LLM translation backend ready: {Path(self._model_path).name}")
        except Exception as exc:
            self._ready = False
            self._state = TranslationState(False, f"LLM backend unavailable: {exc}")

    def _active_message(self) -> str:
        placement = "gpu" if self._effective_gpu_layers > 0 else "cpu"
        detail = "" if not self._owns_server else f"; ctx={self._effective_context}; {placement}"
        return (
            f"LLM translation active: target={self._target_code}; "
            f"model={Path(self._model_path).name}; port={self._port}{detail}"
        )

    # --- auto sizing + retry ladder (round 0075) ---

    def _resolve_auto_sizing(self) -> None:
        if not self._context_auto:
            self._effective_context = self._context_size
            self._effective_gpu_layers = self._gpu_layers
            return
        weights_mib = 2500
        try:
            model = Path(self._model_path)
            # Path('').stat() resolves to the CWD and does NOT raise — only trust a real file.
            if self._model_path and model.is_file():
                weights_mib = int(model.stat().st_size / (1024 * 1024))
        except Exception:
            pass
        free = probe_free_vram_mib()
        if free is None:
            self._effective_context = _CONSERVATIVE_AUTO_CONTEXT
            self._effective_gpu_layers = self._gpu_layers
            self._emit(
                "LLM translation: free-VRAM probe unavailable; using conservative "
                f"ctx={_CONSERVATIVE_AUTO_CONTEXT}."
            )
            return
        budget = free - self._asr_reserve_mib
        ctx, use_gpu = choose_context_tier(budget, weights_mib)
        self._effective_context = ctx
        self._effective_gpu_layers = self._gpu_layers if use_gpu else 0
        self._emit(
            f"LLM translation auto-sizing: free={free}MiB, asr_reserve={self._asr_reserve_mib}MiB, "
            f"budget={budget}MiB, weights~{weights_mib}MiB -> ctx={ctx}, "
            f"{'gpu' if use_gpu else 'CPU fallback'}."
        )

    def _attempt_ladder(self) -> list[tuple[int, int]]:
        """(context, gpu_layers) attempts, strongest first.

        Auto context: halve the context down to the smallest tier, then CPU. A manual
        (pinned) context is NEVER changed — only GPU layers degrade to CPU.
        """
        ctx, ngl = self._effective_context, self._effective_gpu_layers
        attempts: list[tuple[int, int]] = [(ctx, ngl)]
        if self._context_auto:
            step = ctx
            while step > CONTEXT_TIERS[-1]:
                step = max(CONTEXT_TIERS[-1], step // 2)
                if ngl > 0:
                    attempts.append((step, ngl))
            if ngl > 0:
                attempts.append((CONTEXT_TIERS[-1], 0))
        elif ngl > 0:
            attempts.append((ctx, 0))
        deduped: list[tuple[int, int]] = []
        for item in attempts:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _start_with_ladder(self) -> bool:
        for index, (ctx, ngl) in enumerate(self._attempt_ladder()):
            if index:
                self._emit(
                    "LLM translation: llama-server failed to start; retrying with "
                    f"ctx={ctx}, {'gpu' if ngl > 0 else 'CPU'} (attempt {index + 1})."
                )
            self._spawn_server(ctx, ngl)
            if self._wait_healthy():
                self._effective_context, self._effective_gpu_layers = ctx, ngl
                return True
            self._kill_owned()
        return False

    def _spawn_server(self, context_size: int, gpu_layers: int) -> None:
        args = [
            self._server_path,
            "-m", self._model_path,
            "-ngl", str(gpu_layers),
            "--port", str(self._port),
            "-c", str(context_size),
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
        self._emit(
            f"LLM translation: started llama-server (pid={self._proc.pid}, port={self._port}, "
            f"ctx={context_size}, ngl={gpu_layers})."
        )

    def _kill_owned(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and self._owns_server:
            try:
                proc.kill()
            except Exception:
                pass

    # --- runtime self-heal (round 0075) ---

    def _maybe_self_heal(self) -> None:
        """Called when a translate request failed: if our own server died, restart once
        at a degraded tier; a second death disables the backend with a clear message."""
        if not self._owns_server:
            return
        proc = self._proc
        if proc is None or proc.poll() is None:
            return  # server still alive: transient timeout/etc., nothing to heal
        if self._heal_attempted:
            self._ready = False
            self._state = TranslationState(
                False,
                "LLM backend disabled: llama-server crashed again after a degraded "
                "restart; subtitles stay source-only.",
            )
            return
        self._heal_attempted = True
        self._ready = False
        self._state = TranslationState(False, "LLM server crashed; restarting with a smaller footprint.")
        threading.Thread(target=self._self_heal, name="llm-translation-heal", daemon=True).start()

    def _degraded_tier(self) -> tuple[int, int]:
        ctx, ngl = self._effective_context, self._effective_gpu_layers
        if self._context_auto and ctx > CONTEXT_TIERS[-1]:
            return max(CONTEXT_TIERS[-1], ctx // 2), ngl
        return ctx, 0  # pinned context (or already smallest): drop to CPU

    def _self_heal(self) -> None:
        try:
            self._kill_owned()
            ctx, ngl = self._degraded_tier()
            self._effective_context, self._effective_gpu_layers = ctx, ngl
            self._emit(
                f"LLM translation: restarting llama-server after crash with ctx={ctx}, "
                f"{'gpu' if ngl > 0 else 'CPU'}."
            )
            self._spawn_server(ctx, ngl)
            if self._wait_healthy():
                self._ready = True
                self._state = TranslationState(True, self._active_message() + " (recovered)")
                self._emit("LLM translation: recovered after restart.")
            else:
                self._kill_owned()
                self._state = TranslationState(
                    False,
                    "LLM backend disabled: restart after crash failed; subtitles stay source-only.",
                )
        except Exception as exc:
            self._ready = False
            self._state = TranslationState(False, f"LLM backend disabled after crash: {exc}")

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
