"""Translation engine: backend wrapper with an off-thread queue/timeout/retry policy (round 0026).

Translation is best-effort *decoration* of the subtitle — it must never stall the STT loop. Today the
loop calls `backend.translate()` inline on its own thread, so a slow/hanging backend (an LLM/cloud call)
would block subtitle emission. The engine fixes that without changing the call site:

- **Policy disabled (default, `queue_max <= 0`)**: `translate()` is a *direct passthrough* to the
  backend — byte-identical to the pre-0026 inline Argos call. No worker thread, no behavior change.
- **Policy enabled (`queue_max > 0`)**: work runs on a single background worker with a bounded queue.
  The caller submits a job and waits at most `timeout_seconds`; if the backend hasn't answered by then
  it returns `None` (source-only subtitle) and moves on — a hung backend cannot stall the loop. The
  worker applies bounded retry with backoff, and a full queue drops the oldest pending job.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from .base import TranslationBackend, TranslationState


class _Job:
    __slots__ = ("text", "source_code", "result", "done")

    def __init__(self, text: str, source_code: str | None) -> None:
        self.text = text
        self.source_code = source_code
        self.result: Optional[str] = None
        self.done = threading.Event()


_SHUTDOWN = object()


class TranslationEngine:
    """Wrap a `TranslationBackend` with the queue/timeout/retry policy described above."""

    def __init__(
        self,
        backend: TranslationBackend | None,
        *,
        timeout_seconds: float = 0.0,
        max_retries: int = 0,
        queue_max: int = 0,
        retry_backoff_seconds: float = 0.0,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._backend = backend
        self._timeout = max(0.0, float(timeout_seconds or 0.0))
        self._max_retries = max(0, int(max_retries or 0))
        self._queue_max = max(0, int(queue_max or 0))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds or 0.0))
        self._on_status = on_status
        # Async policy is active only when a queue is requested AND there is a backend to call.
        self._async = self._queue_max > 0 and backend is not None
        self._jobs: "queue.Queue[object]" = queue.Queue(maxsize=self._queue_max) if self._async else queue.Queue()
        self._worker: threading.Thread | None = None
        if self._async:
            self._worker = threading.Thread(target=self._run, name="translation-engine", daemon=True)
            self._worker.start()

    # --- backend-mirroring surface (the loop/controller treat the engine like a backend) ---

    @property
    def name(self) -> str:
        return getattr(self._backend, "name", "none")

    @property
    def backend(self) -> TranslationBackend | None:
        return self._backend

    @property
    def policy_active(self) -> bool:
        """True when the off-thread queue/timeout policy is in effect (vs inline passthrough)."""
        return self._async

    @property
    def enabled(self) -> bool:
        return bool(self._backend is not None and getattr(self._backend, "enabled", False))

    @property
    def state(self) -> TranslationState:
        backend_state = getattr(self._backend, "state", None)
        if isinstance(backend_state, TranslationState):
            return backend_state
        return TranslationState(False, "No translation backend.")

    # --- translation ---

    def translate(self, text: str, source_code: str | None = None) -> Optional[str]:
        if not self.enabled:
            return None
        if not self._async:
            # Inline passthrough: identical to the historical direct backend call.
            return self._backend.translate(text, source_code)  # type: ignore[union-attr]
        return self._translate_async(text, source_code)

    def _translate_async(self, text: str, source_code: str | None) -> Optional[str]:
        job = _Job(text, source_code)
        self._enqueue_drop_oldest(job)
        timeout = self._timeout if self._timeout > 0 else None
        if not job.done.wait(timeout):
            # Backend is still working; abandon this result so the loop is never blocked beyond timeout.
            return None
        return job.result

    def _enqueue_drop_oldest(self, job: _Job) -> None:
        """Put a job on the bounded queue, dropping the oldest pending job if it is full."""
        while True:
            try:
                self._jobs.put_nowait(job)
                return
            except queue.Full:
                try:
                    dropped = self._jobs.get_nowait()
                    if isinstance(dropped, _Job):
                        dropped.done.set()  # unblock any waiter on the dropped job (result stays None)
                except queue.Empty:
                    pass

    def _run(self) -> None:
        while True:
            item = self._jobs.get()
            if item is _SHUTDOWN:
                return
            if not isinstance(item, _Job):
                continue
            if item.done.is_set():
                # Dropped/timed-out before we got to it.
                continue
            try:
                item.result = self._call_with_retry(item.text, item.source_code)
            except Exception:
                item.result = None
            finally:
                item.done.set()

    def _call_with_retry(self, text: str, source_code: str | None) -> Optional[str]:
        attempts = self._max_retries + 1
        for index in range(attempts):
            try:
                out = self._backend.translate(text, source_code)  # type: ignore[union-attr]
            except Exception:
                out = None
            if out:
                return out
            if index < attempts - 1 and self._retry_backoff > 0:
                time.sleep(self._retry_backoff * (index + 1))
        return None

    def shutdown(self) -> None:
        """Stop the worker thread (best-effort; safe to call when policy is inactive)."""
        if self._async and self._worker is not None:
            try:
                self._jobs.put_nowait(_SHUTDOWN)
            except queue.Full:
                # Make room and retry once.
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._jobs.put_nowait(_SHUTDOWN)
                except queue.Full:
                    pass


def build_translation_engine(
    config: object,
    *,
    on_status: Optional[Callable[[str], None]] = None,
) -> TranslationEngine:
    """Build the configured backend (via the registry) wrapped in the policy engine.

    Defaults keep the engine in inline-passthrough mode (`translation_queue_max = 0`), so the live path
    is byte-identical to the pre-0026 direct Argos call until the policy is explicitly enabled.
    """
    from .registry import build_backend

    backend = build_backend(getattr(config, "translation_backend", "argos"), config, on_status=on_status)
    return TranslationEngine(
        backend,
        timeout_seconds=getattr(config, "translation_request_timeout_seconds", 0.0),
        max_retries=getattr(config, "translation_max_retries", 0),
        queue_max=getattr(config, "translation_queue_max", 0),
        retry_backoff_seconds=getattr(config, "translation_retry_backoff_seconds", 0.0),
        on_status=on_status,
    )
