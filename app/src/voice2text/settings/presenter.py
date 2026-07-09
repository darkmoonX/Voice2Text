"""Settings presenter helpers: config-to-view-model and option suggestions."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import RuntimeConfig
from ..model_paths import library_model_dir


def loopback_indices_from_config(config: RuntimeConfig) -> list[int]:
    indices = list(config.source_device_indices)
    if not indices and config.device_index is not None:
        indices = [config.device_index]
    return indices


def app_names_from_config(config: RuntimeConfig) -> list[str]:
    names = list(config.source_app_names)
    if not names and config.source_app_name:
        names = [config.source_app_name]
    return names


def normalize_source_language(source_language: str | None) -> str:
    token = source_language or "auto"
    if token == "zh":
        return "zh-hant"
    return token


def normalize_alignment_folder_language(value: str | None) -> str:
    """Fold a language code to the key `whisperx_alignment_model_defaults` is keyed by.

    Mirrors `stt/whisperx_provider.py::_normalize_alignment_folder_language` (duplicated, not
    imported, so the Settings dialog / bootstrap process never pulls in the heavy STT module).
    """
    token = (value or "").strip().lower()
    if not token:
        return ""
    if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
        return "zh"
    return token


def alignment_repos_for_language(source_language: str) -> list[str]:
    token = (source_language or "auto").strip().lower()
    if token in {"zh", "zh-hant", "zh-hans"}:
        return [
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            "wbbbbb/wav2vec2-large-chinese-zh-cn",
            "TencentGameMate/chinese-wav2vec2-base",
        ]
    if token == "ja":
        return [
            "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
            "patrickvonplaten/wav2vec2-large-xlsr-53-japanese",
        ]
    if token == "ko":
        return [
            "kresnik/wav2vec2-large-xlsr-korean",
            "jonatasgrosman/wav2vec2-large-xlsr-53-korean",
        ]
    if token == "en":
        return ["WAV2VEC2_ASR_LARGE_LV60K_960H", "WAV2VEC2_ASR_BASE_960H"]
    return [
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "WAV2VEC2_ASR_BASE_960H",
        "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "wbbbbb/wav2vec2-large-chinese-zh-cn",
        "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "jonatasgrosman/wav2vec2-large-xlsr-53-korean",
    ]


def discover_custom_alignment_candidates(align_root: Path | None = None) -> dict[str, list[str]]:
    """Round 0081: language -> extra custom alignment models found on disk.

    Scans `models/whisperx/align/{hf,custom}/*` for the `.v2t_align_meta.json` sidecar that
    `stt/whisperx_provider.py::_write_alignment_candidate_tag` writes once a repo/bundle has
    actually been loaded for a language. This is what lets the Settings dialog's dropdown
    reflect what's genuinely been downloaded/used, without a separately-persisted registry
    that could drift out of sync with the real cache — read-only, pathlib/json only (no torch/
    whisperx import), safe to call from the Settings dialog / bootstrap process.

    Silently skips anything unreadable/malformed: this is a best-effort discovery layer, not a
    source of truth (`whisperx_alignment_model_defaults` remains that for what's actually
    *applied*), so a corrupt or missing tag should just mean "not discovered," never an error.

    `align_root` overrides the real shared model cache — tests must pass a temp directory here
    rather than relying on the default, since the default touches the same on-disk cache the
    real running app uses (and its contents can change once this feature ships and gains real
    tags, which would otherwise make dropdown-content tests flaky/environment-dependent).
    """
    align_root = align_root if align_root is not None else (library_model_dir("whisperx") / "align")
    discovered: dict[str, list[str]] = {}
    for subdir_name in ("hf", "custom"):
        subdir = align_root / subdir_name
        if not subdir.is_dir():
            continue
        for entry in sorted(subdir.iterdir()):
            if not entry.is_dir():
                continue
            tag_path = entry / ".v2t_align_meta.json"
            if not tag_path.is_file():
                continue
            try:
                payload = json.loads(tag_path.read_text(encoding="utf-8"))
                language = str(payload.get("language") or "").strip()
                model = str(payload.get("model") or "").strip()
            except Exception:
                continue
            if not language or not model:
                continue
            discovered.setdefault(language, [])
            if model not in discovered[language]:
                discovered[language].append(model)
    return discovered
