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

    total_expected_bytes: int | None = None
    baseline_existing_bytes = 0
    try:
        dry_run_items = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir=output_dir,
            dry_run=True,
            token=token,
            revision=revision,
        )
        items = dry_run_items if isinstance(dry_run_items, list) else []
        total_sum = 0
        for item in items:
            file_size = getattr(item, 'size', None)
            file_name = str(getattr(item, 'file_name', '') or '')
            if isinstance(file_size, int) and file_size > 0:
                total_sum += file_size
            if file_name:
                local_path = Path(output_dir) / file_name
                if local_path.exists():
                    try:
                        baseline_existing_bytes += min(int(local_path.stat().st_size), int(file_size) if isinstance(file_size, int) and file_size > 0 else int(local_path.stat().st_size))
                    except Exception:
                        pass
        if total_sum > 0:
            total_expected_bytes = total_sum
    except Exception:
        total_expected_bytes = None

    progress_by_file: dict[str, int] = {}
    size_by_file: dict[str, int] = {}
    last_emitted_percent = -1
    last_emitted_unknown = -1

    class CallbackTqdm(tqdm):

        def __init__(self, *args, **kwargs) -> None:
            kwargs['disable'] = False
            super().__init__(*args, **kwargs)
            self._emit(force=True)

        def update(self, n=1):
            out = super().update(n)
            self._emit()
            return out

        def close(self) -> None:
            self._emit(force=True)
            super().close()

        def _emit(self, force: bool=False) -> None:
            nonlocal last_emitted_percent, last_emitted_unknown
            unit = str(getattr(self, 'unit', '') or '').lower()
            desc = str(getattr(self, 'desc', '') or '')
            current = int(float(self.n))
            total_value = int(float(self.total)) if self.total else None

            # Ignore file-count/progress bookkeeping bars; we only want byte-based bars.
            if unit not in {'b', 'ib', 'bytes'}:
                return

            key = desc or f'file_{id(self)}'
            progress_by_file[key] = max(0, current)
            if total_value and total_value > 0:
                size_by_file[key] = total_value

            running_bytes = baseline_existing_bytes
            for k, v in progress_by_file.items():
                limit = size_by_file.get(k)
                running_bytes += min(v, limit) if (limit and limit > 0) else v

            if total_expected_bytes and total_expected_bytes > 0:
                bounded = min(running_bytes, total_expected_bytes)
                percent = int(bounded * 100 / max(1, total_expected_bytes))
                if not force and percent == last_emitted_percent and bounded != total_expected_bytes:
                    return
                last_emitted_percent = percent
                emit_progress(progress_callback, format_download_progress(provider, model_name, bounded, total_expected_bytes))
            else:
                step = 2 * 1024 * 1024
                if not force and abs(running_bytes - last_emitted_unknown) < step:
                    return
                last_emitted_unknown = running_bytes
                emit_progress(progress_callback, format_download_progress(provider, model_name, running_bytes, None))

    kwargs: dict[str, object] = {'allow_patterns': allow_patterns, 'local_dir': output_dir, 'tqdm_class': CallbackTqdm}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if revision:
        kwargs['revision'] = revision
    if token is not None:
        kwargs['token'] = token
    return str(snapshot_download(repo_id=repo_id, **kwargs))

