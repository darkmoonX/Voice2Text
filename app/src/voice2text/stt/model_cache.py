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

# HuggingFace-hub / kenlm internal folders that are NOT models of their own — never list them as
# entries (they showed up as bogus `.no_exist` / `blobs` / `refs` / `snapshots` / `language_model` rows).
_INTERNAL_DIR_NAMES = {"blobs", "refs", "snapshots", ".no_exist", ".cache", ".locks", "language_model"}
_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".msgpack")


def _is_internal_dir(name: str) -> bool:
    return name in _INTERNAL_DIR_NAMES or name.startswith(".")


def _is_hf_hub_dir(directory: Path) -> bool:
    return directory.name.startswith("models--")


def _pretty_hub_name(name: str) -> str:
    return name[len("models--"):].replace("--", "/") if name.startswith("models--") else name


def _is_model_dir(directory: Path) -> bool:
    """A directory that *directly* holds model files (config.json or a weights file)."""
    try:
        for child in directory.iterdir():
            if not child.is_file():
                continue
            if child.name == "config.json" or child.name.lower().endswith(_WEIGHT_SUFFIXES):
                return True
    except OSError:
        return False
    return False


def _model_dir_size(directory: Path) -> int:
    """Size of a model folder, counting only `blobs/` for HF-hub dirs so symlinked/copied
    `snapshots/` are not double-counted."""
    if _is_hf_hub_dir(directory):
        blobs = directory / "blobs"
        if blobs.is_dir():
            return _dir_size_bytes(blobs)
    return _dir_size_bytes(directory)


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


def _make_entry(directory: Path, *, kind: str, lang: str, name: str | None = None) -> ModelCacheEntry:
    size = _model_dir_size(directory)
    return ModelCacheEntry(
        kind=kind,
        name=name if name is not None else directory.name,
        lang=lang,
        path=str(directory.resolve()),
        size_bytes=size,
        ready=size > 0,
    )


def scan_model_cache(root: str | Path | None = None) -> ModelCacheScan:
    """Enumerate cached base (`stt`) + alignment model folders with sizes/readiness.

    Detects *model* folders (those directly holding `config.json`/weights, or HuggingFace `models--*`
    hub dirs) and skips hub/kenlm internals (`blobs`/`refs`/`snapshots`/`.no_exist`/`.cache`/
    `language_model`). HF-hub sizes count `blobs/` only so symlinked `snapshots/` are not double-counted.
    """
    base = Path(root).resolve() if root is not None else whisperx_cache_root().resolve()
    scan = ModelCacheScan(root=str(base))
    if not base.is_dir():
        return scan

    # Base ASR models: every immediate child dir under stt/ (listed even if empty, so a partial
    # download is visible as not-ready).
    stt_dir = base / "stt"
    if stt_dir.is_dir():
        for child in sorted(stt_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not _is_internal_dir(child.name):
                scan.entries.append(_make_entry(child, kind="stt", lang=""))

    align_root = base / "align"
    for bucket in _ALIGN_SUBDIRS:
        bucket_dir = align_root / bucket
        if not bucket_dir.is_dir():
            continue
        for child in sorted(bucket_dir.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() or _is_internal_dir(child.name):
                continue
            if _is_hf_hub_dir(child):
                scan.entries.append(_make_entry(child, kind="align", lang="", name=_pretty_hub_name(child.name)))
            elif _is_model_dir(child):
                # A flat lang dir whose model files sit at its root (may also have a language_model/ subdir,
                # which is counted in its size, not listed separately).
                scan.entries.append(_make_entry(child, kind="align", lang=child.name))
            else:
                # A container lang dir: list its model subdirs, skip internals/junk.
                lang = child.name
                for sub in sorted(child.iterdir(), key=lambda p: p.name.lower()):
                    if not sub.is_dir() or _is_internal_dir(sub.name):
                        continue
                    if _is_hf_hub_dir(sub):
                        scan.entries.append(_make_entry(sub, kind="align", lang=lang, name=_pretty_hub_name(sub.name)))
                    elif _is_model_dir(sub):
                        scan.entries.append(_make_entry(sub, kind="align", lang=lang))
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
