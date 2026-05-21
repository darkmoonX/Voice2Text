"""Runtime recovery adapter for whisper CUDA failures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class RuntimeRecoveryState:
    attempted_alias_recovery: bool = False
    cpu_fallback_done: bool = False


class WhisperRuntimeRecovery:
    def __init__(
        self,
        *,
        state: RuntimeRecoveryState,
        provider_name: str,
        model_device: str,
        cpu_fallback_on_cuda_error: bool,
        try_prepare_cuda_compat_alias: Callable[[], bool],
        rebuild_transcriber_cuda: Callable[[], object],
        rebuild_transcriber_cpu: Callable[[], object],
        emit_status: Callable[[str], None],
        emit_error: Callable[[str], None],
    ) -> None:
        self._state = state
        self._provider_name = provider_name
        self._model_device = model_device
        self._cpu_fallback_on_cuda_error = cpu_fallback_on_cuda_error
        self._try_prepare_cuda_compat_alias = try_prepare_cuda_compat_alias
        self._rebuild_transcriber_cuda = rebuild_transcriber_cuda
        self._rebuild_transcriber_cpu = rebuild_transcriber_cpu
        self._emit_status = emit_status
        self._emit_error = emit_error

    def try_recover(self, raw_message: str) -> tuple[bool, object | None]:
        if self._provider_name != 'whisper':
            return (False, None)
        if not self._model_device.lower().startswith('cuda'):
            return (False, None)
        if 'cublas64_12.dll' not in raw_message and 'cannot be loaded' not in raw_message:
            return (False, None)

        if not self._state.attempted_alias_recovery:
            self._state.attempted_alias_recovery = True
            self._emit_error('Runtime CUDA DLL error detected. Trying cublas64_13 -> cublas64_12 compatibility alias.')
            if self._try_prepare_cuda_compat_alias():
                try:
                    transcriber = self._rebuild_transcriber_cuda()
                    self._emit_status('CUDA transcriber reloaded after DLL compatibility patch.')
                    return (True, transcriber)
                except Exception as retry_exc:
                    self._emit_error(f'Runtime CUDA retry failed: {retry_exc}')

        if self._cpu_fallback_on_cuda_error and (not self._state.cpu_fallback_done):
            self._state.cpu_fallback_done = True
            try:
                transcriber = self._rebuild_transcriber_cpu()
                self._emit_status('Runtime fallback active: switched to CPU because CUDA DLL could not be loaded.')
                return (True, transcriber)
            except Exception as cpu_exc:
                self._emit_error(f'Runtime CPU fallback failed: {cpu_exc}')

        return (False, None)
