"""Headless model/alignment cache inspection for the WhisperX assets (round 0022).

Pure filesystem helpers — no torch/Qt — so the cache manager UI (Phase B) and the CLI/tests share one
implementation. Scans the cache root (`models/whisperx/...`), reports per-entry sizes + readiness, and
offers a guarded delete that refuses any path outside the cache root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

from ..model_paths import library_model_dir


# Top-level buckets under `models/whisperx/`. `stt` holds base ASR models; `align/*` holds the
# per-language wav2vec2 alignment assets in their HF/torch/cache/custom layouts.
_ALIGN_SUBDIRS = ("hf", "torch", "cache", "custom")


def whisperx_cache_root() -> Path:
    """The WhisperX model/alignment cache root (`app/src/models/whisperx`)."""
    return library_model_dir("whisperx")


def human_size(num_bytes: float) -> str:
    """Human-readable byte size (binary units), e.g. 1536 -> '1.5 KB'."""
    value = float(max(0.0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
    except OSError:
        return total
    return total


def _dir_ready(path: Path) -> bool:
    """A cache dir is 'ready' when it holds at least one non-empty file."""
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file() and child.stat().st_size > 0:
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


@dataclass
class ModelCacheEntry:
    kind: str            # "stt" | "align" | "other"
    name: str            # display name (model folder name)
    lang: str            # alignment language bucket ("" for non-align)
    path: str
    size_bytes: int
    ready: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "lang": self.lang,
            "path": self.path,
            "size_bytes": int(self.size_bytes),
            "size_human": human_size(self.size_bytes),
            "ready": bool(self.ready),
        }


@dataclass
class ModelCacheScan:
    root: str
    entries: list[ModelCacheEntry] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(int(e.size_bytes) for e in self.entries)

    def bucket_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            totals[entry.kind] = totals.get(entry.kind, 0) + int(entry.size_bytes)
        return totals

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "total_bytes": int(self.total_bytes),
            "total_human": human_size(self.total_bytes),
            "bucket_totals": {k: int(v) for k, v in self.bucket_totals().items()},
            "entries": [e.as_dict() for e in self.entries],
        }


def _scan_model_dirs(parent: Path, *, kind: str, lang: str = "") -> list[ModelCacheEntry]:
    entries: list[ModelCacheEntry] = []
    if not parent.is_dir():
        return entries
    for child in sorted(parent.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        entries.append(
            ModelCacheEntry(
                kind=kind,
                name=child.name,
                lang=lang,
                path=str(child.resolve()),
                size_bytes=_dir_size_bytes(child),
                ready=_dir_ready(child),
            )
        )
    return entries


def scan_model_cache(root: str | Path | None = None) -> ModelCacheScan:
    """Enumerate cached base (`stt`) + alignment (`align/<bucket>/<lang>`) model folders with sizes/readiness."""
    base = Path(root).resolve() if root is not None else whisperx_cache_root().resolve()
    scan = ModelCacheScan(root=str(base))
    if not base.is_dir():
        return scan

    # Base ASR models under stt/<model>.
    scan.entries.extend(_scan_model_dirs(base / "stt", kind="stt"))

    # Alignment assets under align/<bucket>/<lang>/<model> (lang layer may be absent in some buckets).
    align_root = base / "align"
    for bucket in _ALIGN_SUBDIRS:
        bucket_dir = align_root / bucket
        if not bucket_dir.is_dir():
            continue
        for lang_dir in sorted(bucket_dir.iterdir(), key=lambda p: p.name.lower()):
            if not lang_dir.is_dir():
                continue
            model_children = [c for c in lang_dir.iterdir() if c.is_dir()]
            if model_children:
                scan.entries.extend(_scan_model_dirs(lang_dir, kind="align", lang=lang_dir.name))
            else:
                # Flat lang dir holding the model files directly.
                scan.entries.append(
                    ModelCacheEntry(
                        kind="align",
                        name=lang_dir.name,
                        lang=lang_dir.name,
                        path=str(lang_dir.resolve()),
                        size_bytes=_dir_size_bytes(lang_dir),
                        ready=_dir_ready(lang_dir),
                    )
                )
    return scan


def cache_summary(root: str | Path | None = None) -> dict[str, object]:
    """Totals-only header for the cache manager (root, total size, per-bucket size, entry count)."""
    scan = scan_model_cache(root)
    return {
        "root": scan.root,
        "total_bytes": int(scan.total_bytes),
        "total_human": human_size(scan.total_bytes),
        "bucket_totals": {k: int(v) for k, v in scan.bucket_totals().items()},
        "entry_count": int(len(scan.entries)),
    }


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def delete_cache_entry(path: str | Path, *, root: str | Path | None = None) -> int:
    """Delete a cache folder, refusing any target outside the cache root. Returns bytes freed.

    Guards against deleting the root itself or anything outside it (a destructive footgun if a UI ever
    passes a stray path). Returns 0 if the path does not exist.
    """
    base = Path(root).resolve() if root is not None else whisperx_cache_root().resolve()
    target = Path(path).resolve()
    if target == base:
        raise ValueError("Refusing to delete the cache root itself.")
    if not _is_within(target, base):
        raise ValueError(f"Refusing to delete a path outside the cache root: {target}")
    if not target.exists():
        return 0
    freed = _dir_size_bytes(target) if target.is_dir() else int(target.stat().st_size)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)
    return int(freed)
