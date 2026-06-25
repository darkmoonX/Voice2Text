"""Settings presenter helpers: config-to-view-model and option suggestions."""
from __future__ import annotations

from ..config import RuntimeConfig


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
        return ["WAV2VEC2_ASR_BASE_960H", "WAV2VEC2_ASR_LARGE_960H"]
    return [
        "WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H",
        "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "wbbbbb/wav2vec2-large-chinese-zh-cn",
        "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "jonatasgrosman/wav2vec2-large-xlsr-53-korean",
    ]
