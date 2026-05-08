from __future__ import annotations

from pathlib import Path


def models_root_dir() -> Path:
    src_dir = Path(__file__).resolve().parent.parent
    root = src_dir / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_model_dir(library_name: str) -> Path:
    normalized = library_name.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("library_name must not be empty")

    target = models_root_dir() / normalized
    target.mkdir(parents=True, exist_ok=True)
    return target
