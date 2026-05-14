"""Preset model metadata and download/extract helpers for Vosk and Sherpa-ONNX."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Callable
import zipfile
from .model_download import download_to_file, emit_progress, format_download_progress

@dataclass(frozen=True)
class ModelPreset:
    provider: str
    key: str
    archive_url: str
    archive_format: str
    target_dir_name: str
    aliases: tuple[str, ...]
    archive_dir_name: str | None = None
    legacy_dir_names: tuple[str, ...] = ()
_MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        provider='vosk',
        key='small-en-us',
        archive_url='https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        archive_format='zip',
        target_dir_name='vosk-model-small-en-us-0.15',
        aliases=('small', 'small-en', 'small-en-us', 'en-us', 'vosk-model-small-en-us-0.15'),
    ),
    ModelPreset(
        provider='vosk',
        key='cn-0.22',
        archive_url='https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip',
        archive_format='zip',
        target_dir_name='vosk-model-cn-0.22',
        aliases=('full-cn', 'zh-cn', 'cn-full', 'vosk-model-cn-0.22'),
    ),
    ModelPreset(
        provider='vosk',
        key='small-zh',
        archive_url='https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip',
        archive_format='zip',
        target_dir_name='vosk-model-small-cn-0.22',
        aliases=('small-zh', 'small-cn', 'zh', 'cn', 'small-cn-0.22', 'vosk-model-small-cn-0.22'),
    ),
    ModelPreset(
        provider='sherpa-onnx',
        key='paraformer-zh',
        archive_url='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2',
        archive_format='tar.bz2',
        target_dir_name='sherpa-onnx-paraformer-zh-2023-03-28',
        aliases=('small', 'paraformer-zh', 'paraformer'),
        archive_dir_name='sherpa-onnx-paraformer-zh-2023-03-28',
        legacy_dir_names=('paraformer-zh-2023-03-28',),
    ),
    ModelPreset(
        provider='sherpa-onnx',
        key='zipformer-zh-en',
        archive_url='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2',
        archive_format='tar.bz2',
        target_dir_name='sherpa-onnx-zipformer-zh-en-2023-11-22',
        aliases=('zipformer-zh-en', 'zh-en'),
        archive_dir_name='sherpa-onnx-zipformer-zh-en-2023-11-22',
        legacy_dir_names=('zipformer-zh-en-2023-11-22',),
    ),
)

def has_model_preset(provider: str, model_ref: str) -> bool:
    return find_model_preset(provider, model_ref) is not None

def find_model_preset(provider: str, model_ref: str) -> ModelPreset | None:
    normalized_provider = provider.strip().lower()
    normalized_ref = model_ref.strip().lower()
    if not normalized_ref:
        return None
    provider_presets = [preset for preset in _MODEL_PRESETS if preset.provider == normalized_provider]
    for preset in provider_presets:
        if normalized_ref == preset.key or normalized_ref == preset.target_dir_name.lower():
            return preset
    for preset in provider_presets:
        if normalized_ref in preset.aliases:
            return preset
    return None

def ensure_model_preset_downloaded(provider: str, model_ref: str, model_root: Path, timeout_seconds: int=180, progress_callback: Callable[[str], None] | None=None) -> Path | None:
    preset = find_model_preset(provider, model_ref)
    if preset is None:
        return None
    target_dir = model_root / preset.target_dir_name
    if _is_model_dir_ready(preset, target_dir):
        emit_progress(progress_callback, f'{provider} model ready: {preset.target_dir_name}')
        return target_dir
    if target_dir.exists():
        emit_progress(progress_callback, f'{provider} model folder exists but looks incomplete, re-downloading: {preset.target_dir_name}')
        shutil.rmtree(target_dir, ignore_errors=True)
    for legacy_name in preset.legacy_dir_names:
        legacy_dir = model_root / legacy_name
        if not legacy_dir.exists():
            continue
        model_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(legacy_dir), str(target_dir))
            emit_progress(progress_callback, f'{provider} model migrated to: {target_dir.name}')
            return target_dir
        except Exception:
            return legacy_dir
    model_root.mkdir(parents=True, exist_ok=True)
    download_root = model_root / '.downloads'
    download_root.mkdir(parents=True, exist_ok=True)
    archive_file = download_root / f'{preset.provider}-{preset.target_dir_name}.archive'
    archive_part_file = Path(str(archive_file) + '.part')
    tmp_root = Path(tempfile.mkdtemp(prefix='voice2text-model-'))
    extract_root = tmp_root / 'extract'
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        emit_progress(progress_callback, f'Start downloading {provider} model: {preset.target_dir_name}')
        download_to_file(preset.archive_url, archive_part_file, timeout_seconds=timeout_seconds, progress_callback=lambda downloaded, total: emit_progress(progress_callback, format_download_progress(provider, preset.target_dir_name, downloaded, total)), resume=True)
        if archive_file.exists():
            archive_file.unlink(missing_ok=True)
        archive_part_file.replace(archive_file)
        emit_progress(progress_callback, f'{provider} model download completed. Extracting: {preset.target_dir_name}')
        _extract_archive(archive_file, extract_root, archive_format=preset.archive_format)
        source_dir = _locate_extracted_model_dir(extract_root, preset.archive_dir_name or preset.target_dir_name)
        if source_dir is None:
            emit_progress(progress_callback, f'{provider} model extracted but model folder not found: {preset.target_dir_name}')
            return None
        if target_dir.exists():
            return target_dir
        shutil.move(str(source_dir), str(target_dir))
        archive_file.unlink(missing_ok=True)
        emit_progress(progress_callback, f'{provider} model prepared: {target_dir.name}')
        return target_dir
    except Exception as exc:
        emit_progress(progress_callback, f'{provider} model download failed: {exc}')
        return None
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

def _extract_archive(archive_path: Path, extract_dir: Path, archive_format: str) -> None:
    if archive_format == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(extract_dir)
        return
    mode = 'r'
    if archive_format == 'tar.bz2':
        mode = 'r:bz2'
    elif archive_format == 'tar.gz':
        mode = 'r:gz'
    with tarfile.open(archive_path, mode) as archive:
        archive.extractall(extract_dir)

def _locate_extracted_model_dir(extract_root: Path, expected_name: str) -> Path | None:
    expected_lower = expected_name.lower()
    direct = extract_root / expected_name
    if direct.is_dir():
        return direct
    candidates: list[Path] = []
    for item in extract_root.rglob('*'):
        if not item.is_dir():
            continue
        if item.name.lower() == expected_lower:
            return item
        candidates.append(item)
    if candidates:
        candidates.sort(key=lambda value: len(value.parts))
        return candidates[0]
    return None

def _is_model_dir_ready(preset: ModelPreset, model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if preset.provider == 'sherpa-onnx':
        tokens = model_dir / 'tokens.txt'
        has_transducer = all(((model_dir / name).exists() for name in ('encoder.onnx', 'decoder.onnx', 'joiner.onnx')))
        has_paraformer = (model_dir / 'model.onnx').exists() or (model_dir / 'model.int8.onnx').exists()
        return tokens.exists() and (has_transducer or has_paraformer)
    if preset.provider == 'vosk':
        has_conf = (model_dir / 'conf' / 'model.conf').exists()
        has_model_file = (model_dir / 'am' / 'final.mdl').exists() or (model_dir / 'graph').is_dir()
        return has_conf or has_model_file
    try:
        next(model_dir.iterdir())
        return True
    except StopIteration:
        return False
