"""Low-level download and progress-format utilities shared by STT model acquisition flows."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import urllib.request
from urllib.error import HTTPError

def emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(message)
    except Exception:
        return

def format_download_progress(provider: str, model_name: str, downloaded: int, total: int | None) -> str:
    bar_width = 28
    downloaded_mb = downloaded / (1024 * 1024)
    if total and total > 0:
        total_mb = total / (1024 * 1024)
        percent = min(100.0, downloaded * 100.0 / total)
        filled = int(round(percent / 100.0 * bar_width))
        filled = max(0, min(bar_width, filled))
        bar = '#' * filled + '-' * (bar_width - filled)
        return f'[download] {provider} downloading: {model_name} [{bar}] {percent:.0f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)'
    cursor = int(downloaded_mb) % bar_width
    bar = '-' * cursor + '>' + '-' * max(0, bar_width - cursor - 1)
    return f'[download] {provider} downloading: {model_name} [{bar}] ({downloaded_mb:.1f} MB)'

def download_to_file(url: str, target_file: Path, timeout_seconds: int, progress_callback: Callable[[int, int | None], None] | None=None, resume: bool=True) -> None:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    resume_offset = 0
    if resume and target_file.exists():
        try:
            resume_offset = max(0, int(target_file.stat().st_size))
        except Exception:
            resume_offset = 0
    request_headers = {'User-Agent': 'Voice2Text/1.0'}
    if resume_offset > 0:
        request_headers['Range'] = f'bytes={resume_offset}-'
    request = urllib.request.Request(url, headers=request_headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        if exc.code == 416 and target_file.exists():
            if progress_callback is not None:
                progress_callback(resume_offset, resume_offset)
            return
        raise
    with response:
        total_size: int | None = None
        append_mode = False
        status = getattr(response, 'status', None)
        content_range = response.headers.get('Content-Range')
        if resume_offset > 0 and (status == 206 or content_range):
            append_mode = True
            if content_range and '/' in content_range:
                try:
                    total_size = int(content_range.rsplit('/', 1)[1])
                except Exception:
                    total_size = None
        if total_size is None:
            length_header = response.headers.get('Content-Length')
            try:
                parsed = int(length_header) if length_header else 0
                if parsed > 0:
                    total_size = parsed + resume_offset if append_mode else parsed
            except Exception:
                total_size = None
        if not append_mode and target_file.exists():
            target_file.unlink(missing_ok=True)
            resume_offset = 0
        downloaded = resume_offset
        next_percent = 1
        next_report_bytes = 2 * 1024 * 1024
        with target_file.open('ab' if append_mode else 'wb') as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback is None:
                    continue
                if total_size:
                    percent = int(downloaded * 100 / total_size)
                    if percent >= next_percent or downloaded >= total_size:
                        progress_callback(downloaded, total_size)
                        while next_percent <= percent:
                            next_percent += 1
                elif downloaded >= next_report_bytes:
                    progress_callback(downloaded, None)
                    next_report_bytes = downloaded + 2 * 1024 * 1024
        if progress_callback is not None:
            progress_callback(downloaded, total_size)

def download_hf_snapshot_with_progress(*, repo_id: str, output_dir: str, allow_patterns: list[str], progress_callback: Callable[[str], None] | None, provider: str, model_name: str, cache_dir: Optional[str]=None, revision: Optional[str]=None, token: Optional[str | bool]=None) -> str:
    try:
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm
    except Exception as exc:
        raise RuntimeError('huggingface_hub/tqdm is required for model download progress') from exc

    class CallbackTqdm(tqdm):

        def __init__(self, *args, **kwargs) -> None:
            kwargs['disable'] = False
            super().__init__(*args, **kwargs)
            self._last_percent = -1
            self._last_unknown_bytes = -1
            self._emit()

        def update(self, n=1):
            out = super().update(n)
            self._emit()
            return out

        def close(self) -> None:
            self._emit(force=True)
            super().close()

        def _emit(self, force: bool=False) -> None:
            current = int(float(self.n))
            total_value = int(float(self.total)) if self.total else None
            if total_value:
                percent = int(current * 100 / max(1, total_value))
                if not force and percent == self._last_percent and (current != total_value):
                    return
                self._last_percent = percent
            else:
                step = 2 * 1024 * 1024
                if not force and abs(current - self._last_unknown_bytes) < step:
                    return
                self._last_unknown_bytes = current
            emit_progress(progress_callback, format_download_progress(provider, model_name, current, total_value))
    kwargs: dict[str, object] = {'allow_patterns': allow_patterns, 'local_dir': output_dir, 'tqdm_class': CallbackTqdm}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if revision:
        kwargs['revision'] = revision
    if token is not None:
        kwargs['token'] = token
    return str(snapshot_download(repo_id=repo_id, **kwargs))
