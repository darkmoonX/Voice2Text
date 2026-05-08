from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

_DLL_DIR_HANDLES: list[object] = []


def ensure_cublas12_from_source(
    source_dll: str,
    compat_dir: Optional[str] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> bool:
    if _can_load_cublas12():
        return True

    source = Path(source_dll)
    if not source.is_file():
        _emit(on_status, f"CUDA compatibility source DLL not found: {source}")
        return False

    target_root = (
        Path(compat_dir)
        if compat_dir
        else (Path(__file__).resolve().parent.parent / "runtime_bin")
    )
    target_root.mkdir(parents=True, exist_ok=True)

    target = target_root / "cublas64_12.dll"

    try:
        if (
            not target.exists()
            or target.stat().st_size != source.stat().st_size
            or target.stat().st_mtime < source.stat().st_mtime
        ):
            shutil.copy2(source, target)
            _emit(on_status, f"Created CUDA DLL alias: {target}")
    except Exception as exc:
        _emit(on_status, f"Failed to prepare CUDA DLL alias: {exc}")
        return False

    _register_dll_search_path(str(source.parent))
    _register_dll_search_path(str(target_root))

    path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{target_root};{source.parent};{path}"

    if _can_load_cublas12():
        return True

    _emit(
        on_status,
        "CUDA compatibility alias exists but cublas64_12.dll still cannot be loaded.",
    )
    return False


def _can_load_cublas12() -> bool:
    try:
        ctypes.WinDLL("cublas64_12.dll")
        return True
    except Exception:
        return False


def _register_dll_search_path(path: str) -> None:
    if not path or not os.path.isdir(path):
        return

    try:
        add_dir = getattr(os, "add_dll_directory", None)
        if add_dir is None:
            return
        handle = add_dir(path)
        _DLL_DIR_HANDLES.append(handle)
    except Exception:
        return


def _emit(on_status: Optional[Callable[[str], None]], message: str) -> None:
    if on_status is not None:
        on_status(message)